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
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, t):
    x_proj = t[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


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
class Dense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
        """Initialize a time-dependent score-based network.

        Args:
        marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
        channels: The number of channels for feature maps of each resolution.
        embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim))
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False, padding=1)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False, padding=1)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False, padding=1)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False, padding=1)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2, bias=False, output_padding=0)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 2, stride=2, bias=False, output_padding=0)    
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 2, stride=2, bias=False, output_padding=0)    
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 1, stride=1, output_padding=0)
        
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
  
    def forward(self, x, t): 
        # Obtain the Gaussian random feature embedding for t   
        embed = self.act(self.embed(t))    
        # Encoding path
        h1 = self.conv1(x)    
        ## Incorporate information from t
        h1 += self.dense1(embed)
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h


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
            noise_norm = np.sqrt(np.prod(x.shape[1:])) # 56
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
    scoreNet = ScoreNet(marginal_prob_std)
    scoreNet = scoreNet.to(DEVICE)
    scoreNet_optimizer = torch.optim.Adam(scoreNet.parameters(), lr = 1e-4)

    epochs = 10
    for i in range(epochs):
        loss = []
        for i_batch, feed_dict in enumerate(tqdm.tqdm(data_loader)):
            x, y = feed_dict
            x = x.to(DEVICE)
            scoreNet_loss = loss_fn(scoreNet, x, marginal_prob_std = marginal_prob_std)
            loss.append(scoreNet_loss.cpu().detach().numpy())
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
    # scoreNet = train_scoreNet(data_loader, batch_size, 56, 56)


    dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)    
    # resize_img = transforms.Resize((56, 56))
    dataset_resize = []
    n_images = 5000
    cnt = 0
    for x, y in dataset:
        cnt += 1
        im = x.cpu().detach().numpy()
        im_rep = im.repeat(2, axis=1).repeat(2, axis=2)
        dataset_resize.append([im_rep, y])
        if cnt > n_images:
            break;
    data_loader = DataLoader(dataset_resize[:n_images], batch_size=BATCH_SIZE, shuffle=True)
    scoreNet = train_scoreNet(data_loader, BATCH_SIZE, 56, 56)

    ve_model = VE_SDE(BATCH_SIZE, 56, 56, scoreNet=scoreNet)    
    
    t = torch.ones(BATCH_SIZE) # initial time이라 1을 넣는가봄
    std = marginal_prob_std(t)[:, None, None, None]
    x = torch.rand((32, 56, 56, 1)) * std

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
    resize_img = transforms.Resize((56, 56))
    dataset_resize = []
    for x, y in dataset:
        x = resize_img(x)
        dataset_resize.append([x, y])
    data_loader = DataLoader(dataset_resize[:5], batch_size=batch_size, shuffle=True)
    scoreNet = train_scoreNet(data_loader, batch_size, 56, 56)

    random_t = torch.rand(x.shape[0], device=DEVICE)
    score = scoreNet(x[None, :, :, :], random_t)
    plot(score)



unit_test_ve_sde()
# unit_test_scorenet()
