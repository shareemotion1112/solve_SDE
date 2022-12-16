import torch
import torch.nn as nn
import torch.nn.functional as F
from Contant import DEVICE
import numpy as np
from cPrint import pp
from Utils import convert_to_torch_tensor, plot
from tqdm import trange

SIGMA = torch.tensor(0.05)
TIME_STEP = 0.01


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
    z = torch.randint(0, 255, x.size(), device=DEVICE)
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
        x_embed = self.embed(x); pp(f"x_embed : {x_embed.shape}")

        self.down1 = DownSample(x_embed.shape[1], self.channels[0])

        x_down1 = self.down1(x_embed) 
        pp(f"x_down1 : {x_down1.size()}")
        x_down2 = self.down2(x_down1) 
        pp(f"x_down2 : {x_down2.size()}")

        x_bottom = self.bottom(x_down2)
        pp(f"x_bottom : {x_bottom.size()}")
        btm = nn.ConvTranspose2d(x_bottom.shape[1], x_bottom.shape[1], kernel_size=2, stride=2, padding=0).to(DEVICE)
        x_bottom = btm(x_bottom)
        pp(f"x_bottom_conv : {x_bottom.size()}")

        x_up1 = self.up1(x_down2, x_bottom); pp(f"x_up1 : {x_up1.shape}")
        x_up2 = self.up2(x_down1, x_up1); pp(f"x_up2 : {x_up2.shape}")
        nnConv2d = nn.Conv2d(x_up2.shape[1], self.n_channel, kernel_size=1, stride=1, padding=0).to(DEVICE)
        x = nnConv2d(x_up2)
        pp(f"x : {x.shape}")
        x_act = self.act(x)
        pp(f"x_act : {x_act.shape}")

        denominator = marginal_prob_std(t)
        pp(f"t : {t}")
        pp(f"denominator : {denominator}")
        # pp(f"denominator : {denominator.shape}")
        pp(f"x_act : {x_act.shape}")
        
        # x = x_act / denominator[:, None, None, None]
        x = x_act / denominator
        # pp(f"final x : {x.shape}")
        
        return x 


class VE_SDE:
    def __init__(self, n_batch, width, height, predictor_steps = 100, corrector_steps = 10, scoreNet = None):
        self.scoreNet = scoreNet
        self.predictor_steps = predictor_steps
        self.corrector_steps = corrector_steps
        self.drift_coef = self.drift_func
        self.diffusion_coef = 0
        self.n_batch = n_batch
        self.width = width
        self.height = height
        self.epsilons = torch.randn(self.corrector_steps)

    def sigma_func(self, t):
        return torch.tensor(t ** 2)

    def drift_func(self, t):
        return torch.sqrt(2 * t) * torch.randn(1)

    def predictor(self, x):
        for t in trange(self.predictor_steps - 1, 0, -1):
            t = t * TIME_STEP
            sigma_diff = (self.sigma_func(t + 1)**2 - self.sigma_func(t)**2)
            x_i_prime = x + sigma_diff * self.scoreNet(x, t)
            z = torch.randn(1).to(DEVICE) # 이거 평균이 0이고 표준편차가 1인 identity matrix인지 확인 필요 : checked!
            x = x_i_prime + torch.sqrt(sigma_diff) * z
        return x
    def corrector(self, x):
        for j in trange(0, self.corrector_steps, 1):
            t = (j + 1) * TIME_STEP # t = 0이면 scoreNet 게산할때 에러남
            z = torch.randn(1).to(DEVICE)

            x = x + self.epsilons[j] * self.scoreNet(x, t) + torch.sqrt(torch.abs(2 * self.epsilons[j])) * z

            if torch.sum(torch.isnan(x)) > 0:
                print(x)
        return x

    def run_denoising(self, x):
        x = self.predictor(x)
        x = self.corrector(x)
        return x

    def run_predictor_only(self, x):
        return self.predictor(x)

def train_scoreNet(data_loader, batch_size, width, height):
    scoreNet = ScoreNet2D(batch_size, 1, width, height)
    scoreNet = scoreNet.to(DEVICE)
    scoreNet_optimizer = torch.optim.Adam(scoreNet.parameters(), lr = 1e-4)

    epochs = 100

    for x, y in data_loader:
        x = x.to(DEVICE)
        for i in trange(epochs):
            scoreNet_loss = loss_fn(scoreNet, x, marginal_prob_std = marginal_prob_std)
            scoreNet_optimizer.zero_grad()
            scoreNet_loss.backward()
            scoreNet_optimizer.step()
    return scoreNet


def unit_test_ve_sde():
    import os
    from ImageHandle import get_img_dataloader
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    from torchvision.datasets import MNIST

    base_dir = "/Users/shareemotion/Projects/Solve_SDE/Data"
    batch_size = 1
    predictor_steps = 50 # 너무 노이즈를 많이 넣어도 학습이 안될 듯
    corrector_steps = 50
    # n_channel = 1
    train_dir = os.path.join(base_dir,'train')
    # test_dir = os.path.join(base_dir,'test1')


    # file_names = os.listdir(train_dir)[:1]
    # data_loader = get_img_dataloader(train_dir, file_names, batch_size)
    # scoreNet = train_scoreNet(data_loader, batch_size, 400, 400)


    dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
    resize_img = transforms.Resize((400, 400))
    dataset_resize = []
    for x, y in dataset:
        x = resize_img(x)
        dataset_resize.append([x, y])
    data_loader = DataLoader(dataset_resize[:5], batch_size=batch_size, shuffle=True)
    scoreNet = train_scoreNet(data_loader, batch_size, 400, 400)

    ve_model = VE_SDE(batch_size, 400, 400, scoreNet=scoreNet, predictor_steps = predictor_steps, corrector_steps=corrector_steps)
    for x, y in data_loader:
        denoising_x = ve_model.run_denoising(x)
        pp("denoising x : {denoising_x.shape}")
        plot(x, scoreNet(x, 1), denoising_x)

        # predictor_x = ve_model.run_predictor_only(x)
        # plot(x, scoreNet(x, 1), predictor_x, denoising_x)
    

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



# unit_test_ve_sde()
# unit_test_scorenet()
