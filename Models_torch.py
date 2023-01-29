import torch
import torch.nn as nn
import torch.nn.functional as F
from Contant import DEVICE
import numpy as np
from cPrint import pp
from Utils import convert_to_torch_tensor, plot, plot_imgs
import tqdm 
import matplotlib.pyplot as plt


SIGMA = torch.tensor(25.)
BATCH_SIZE = 32





class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, scale=30.):
    super().__init__()
    self.W = None
    self.scale = 30
  def forward(self, x):
    """
    x : [n_batch, n_channel, width, height]
    """        
    x = x.to(DEVICE)
    self.W = nn.Parameter(torch.randn(x.shape[2], x.shape[3]) * self.scale, requires_grad=False).to(DEVICE)
    x_proj = torch.zeros(x.shape[0], x.shape[1] * 2, x.shape[2], x.shape[3]).to(DEVICE)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp = x[i, j, :, :] * self.W * 2 * np.pi
            m_cat = torch.cat((torch.sin(tmp)[None, :, :], torch.cos(tmp)[None, :, :]), dim=0)
            x_proj[i, j:(j+2), :, :] = m_cat
    return x_proj

def marginal_prob_std(t, sigma = SIGMA):
    if type(t) != torch.TensorType:
        t = torch.tensor(t)
    t = t.clone()
    result = torch.sqrt((sigma**(2 * t) - 1.) / 2. / torch.log(sigma))
    return result


def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    random_t = torch.rand(x.shape[0], device=DEVICE) * (1. - eps) + eps
    # pp(x.size())
    z = torch.randn(x.size(), device=DEVICE)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]
    score = model(perturbed_x, random_t)

    pp(f"perturbed x : {perturbed_x.shape}, score : {score.shape}, std : {std.shape}, z : {z.shape}") 
    pp(f"{(score * std[:, None, None, None] + z)**2}")
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
    return loss


"""
ScoreNet: input의 ㅣog probability를 학습하는 녀석
channels, stride 설명 : 칼라갯수 또는 예측하고자 하는 클래스 수, filter가 이동하는 거리
http://taewan.kim/post/cnn/
https://justkode.kr/deep-learning/pytorch-cnn

Group Normalization :  batch의 크기가 극도로 작은 상황에서 batch normalization 대신 사용하면 좋은 normalization 기술
LN과 IN의 절충된 형태로 볼 수 있는데, 각 채널을 N개의 group으로 나누어 normalize 시켜주는 기술입니다.
https://blog.lunit.io/2018/04/12/group-normalization/

ConvTranspose2d : https://cumulu-s.tistory.com/29

"""


class DownSample(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1).to(DEVICE)
        self.gn = nn.GroupNorm(output_dim, output_dim).to(DEVICE)
        self.relu = nn.ReLU().to(DEVICE)
        self.pool = nn.MaxPool2d(kernel_size=2).to(DEVICE)
    def forward(self, x):
        a = x.device
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class UpSample(nn.Module):
    """
    두배로 늘어난 채널을 다시 반으로 감소 시켜야함
    """
    def __init__(self):
        super(UpSample, self).__init__()
        self.input_dim = 12
        self.gn = None

    def tconv(self, n_ch):
        return nn.ConvTranspose2d(n_ch, n_ch, kernel_size=2, stride=2, padding=0).to(DEVICE)

    def conv(self, n_ch):
        return nn.Conv2d(n_ch, n_ch // 2, kernel_size=3, stride=1, padding=1).to(DEVICE)
        
    def forward(self, previous_x, x):
        x = self.conv(x.shape[1])(x)
        pp(f"shape check : {previous_x.shape} vs {x.shape}")
        x = torch.cat([previous_x, x], dim=1); pp(f"x : {x.shape}")        
        x = self.conv(x.shape[1])(x)
        x = self.tconv(x.shape[1])(x, output_size=(previous_x.shape[2] * 2, previous_x.shape[3] * 2))
        groupNorm = nn.GroupNorm(x.shape[1], x.shape[1]).to(DEVICE)
        x = groupNorm(x)
        return x    

class ScoreNet2D(nn.Module):
    def __init__(self, n_batch, n_channel, width, height, channels=[6, 12, 24, 48]):
        super(ScoreNet2D, self).__init__()
        
        self.n_batch = n_batch
        self.n_channel = n_channel
        self.width = width
        self.height = height
        self.channels = channels
        
        self.embed = GaussianFourierProjection()

        self.down2 = DownSample(channels[0], channels[1])
        self.bottom = DownSample(channels[1], channels[2])

        self.up1 = UpSample()
        self.up2 = UpSample()

        self.dense1_input_size = 1        
        self.dense1 = nn.Linear(self.dense1_input_size, n_batch * n_channel * width * height)

        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std


    def forward(self, x, t): # t는 loss_fn에 정의된 random_t가 들어감
        if type(x) == np.ndarray:
            x = convert_to_torch_tensor(x)
        x_embed = self.embed(x)

        self.down1 = DownSample(x_embed.shape[1], self.channels[0])

        x_down1 = self.down1(x_embed) 
        x_down2 = self.down2(x_down1) 

        x_bottom = self.bottom(x_down2)
        btm = nn.ConvTranspose2d(x_bottom.shape[1], x_bottom.shape[1], kernel_size=2, stride=2, padding=0).to(DEVICE)
        x_bottom = btm(x_bottom)

        x_up1 = self.up1(x_down2, x_bottom)
        x_up2 = self.up2(x_down1, x_up1)
        nnConv2d = nn.Conv2d(x_up2.shape[1], self.n_channel, kernel_size=1, stride=1, padding=0).to(DEVICE)
        x = nnConv2d(x_up2)
        x_act = self.act(x)

        denominator = marginal_prob_std(t)[:, None, None, None]
        x = x_act / denominator        
        return x 


class VE_SDE:
    def __init__(self, n_batch, width, height, scoreNet = None):
        self.scoreNet = scoreNet
        self.n_batch = n_batch
        self.width = width
        self.height = height
        self.epsilon = torch.tensor(1e-5)

    def diffusion_coef(self, t, sigma=SIGMA):
        return sigma ** t
    
    def run_pc_sampler(self, x, num_steps=500, eps=1e-3, snr=0.16):
        time_steps = np.linspace(1., eps, num_steps)
        step_size = time_steps[0] - time_steps[1]
        
        for i, time_step in enumerate(tqdm(time_steps)):
            batch_time_step = torch.ones(self.n_batch) * time_step

            # corrector step (Langevin MCMC)
            grad = self.scoreNet(x, batch_time_step)            
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:])) # 400
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + torch.sqrt(2*langevin_step_size) * torch.random.randn(x.shape)

            # predictor step (Euler-Maruyama)
            g = self.diffusion_coef(batch_time_step)
            x_mean = x + (g**2)[:, None, None, None] * self.scoreNet(x, batch_time_step) * step_size
            x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.random.randn(x.shape)
            if i % 10 == 0:
                plot_imgs(x)
                plt.pause(0.1)
                plt.close()   
        # The last step does not include any noise !!!!!!
        return x_mean

def train_scoreNet(data_loader, batch_size, width, height):
    scoreNet = ScoreNet2D(batch_size, 1, width, height)
    scoreNet = scoreNet.to(DEVICE)
    scoreNet_optimizer = torch.optim.Adam(scoreNet.parameters(), lr = 1e-4)

    epochs = 10
    for i in range(epochs):
        loss = []
        for i_batch, feed_dict in enumerate(tqdm.tqdm(data_loader)):
            x, y = feed_dict
            x = x.to(DEVICE)
            scoreNet_loss = loss_fn(scoreNet, x, marginal_prob_std = marginal_prob_std)
            loss.append(scoreNet_loss.detach().numpy())
            scoreNet_optimizer.zero_grad()
            scoreNet_loss.backward()
            scoreNet_optimizer.step()
        print(f"epochs : {i}, loss : {np.mean(loss)}")
    return scoreNet


def unit_test_ve_sde():
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    from torchvision.datasets import MNIST

    # n_channel = 1
    # train_dir = os.path.join(base_dir,'train')
    # test_dir = os.path.join(base_dir,'test1')


    # file_names = os.listdir(train_dir)[:1]
    # data_loader = get_img_dataloader(train_dir, file_names, batch_size)
    # scoreNet = train_scoreNet(data_loader, batch_size, 400, 400)


    dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)    
    resize_img = transforms.Resize((400, 400))
    dataset_resize = []
    n_images = 5000
    cnt = 0
    for x, y in dataset:
        cnt += 1
        x = resize_img(x)
        dataset_resize.append([x, y])
        if cnt > n_images:
            break;
    data_loader = DataLoader(dataset_resize[:n_images], batch_size=BATCH_SIZE, shuffle=True)
    scoreNet = train_scoreNet(data_loader, BATCH_SIZE, 400, 400)

    ve_model = VE_SDE(BATCH_SIZE, 400, 400, scoreNet=scoreNet)    
    
    t = torch.ones(BATCH_SIZE) # initial time이라 1을 넣는가봄
    std = marginal_prob_std(t)[:, None, None, None]
    x = torch.random.randn((32, 400, 400, 1)) * std

    denoised_x = ve_model.run_pc_sampler(x)
    plot_imgs(denoised_x)


    # # 데이터의 가운데를 지우고 테스트 
    # offset = 50
    # for x, y in data_loader:
    #     x_cp = x.clone()
    #     x_cp[:, :, (200-offset):(200+offset), (200-offset):(200+offset)] = 0
    #     denoising_x = ve_model.run_pc_sampler(x_cp)
    #     plot(x, scoreNet(x, 1), denoising_x, name=["origin", "score", "denoised"])


def unit_test_scorenet():    
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    from torchvision.datasets import MNIST

    batch_size = 1

    dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
    resize_img = transforms.Resize((400, 400))
    dataset_resize = []
    for x, y in dataset:
        x = resize_img(x)
        dataset_resize.append([x, y])
    data_loader = DataLoader(dataset_resize[:5], batch_size=batch_size, shuffle=True)
    scoreNet = train_scoreNet(data_loader, batch_size, 400, 400)

    random_t = torch.rand(x.shape[0], device=DEVICE)
    score = scoreNet(x[None, :, :, :], random_t)
    plot(score)



unit_test_ve_sde()
# unit_test_scorenet()
