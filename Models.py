import torch
import torch.nn as nn
import torch.nn.functional as F
from Contant import DEVICE
import numpy as np

verbose = True

def pp(txt):
    if verbose is True:
        print(txt)

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    print(f"W size : {self.W.size()}")
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

def marginal_prob_std(t, sigma):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:    
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.  

    Returns:
    The standard deviation.
    """    
    t = torch.tensor(t, device=DEVICE)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    """The loss function for training score-based generative models.

    Args:
    model: A PyTorch model instance that represents a 
        time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
        the perturbation kernel.
    eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  # 왜 random T를 사용? int 스텝을 넣는게 아니네??
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]
    score = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
    return loss

class NeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256, device=DEVICE)
        self.fc2 = nn.Linear(256, 128, device=DEVICE)
        self.fc3 = nn.Linear(128, 64, device=DEVICE)
        self.fc4 = nn.Linear(64, output_dim, device=DEVICE)

    def forward(self, x):
        x = nn.BatchNorm1d(self.fc1(x))
        x = nn.BatchNorm1d(self.fc2(x))
        x = nn.BatchNorm1d(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

"""
ScoreNet: input의 ㅣog probability를 학습하는 녀석
channels, stride 설명 : http://taewan.kim/post/cnn/
https://justkode.kr/deep-learning/pytorch-cnn
"""

class ScoreNet2D(nn.Module):
    def __init__(self, in_channel = 1, n_channel = 1, kernel_size=3, embed_dim=256):
        super(ScoreNet2D, self).__init__()
        self.embed_dim = embed_dim
        self.n_channel = n_channel
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim), \
         nn.Linear(embed_dim, embed_dim))
        self.conv1 = nn.Conv2d(in_channel, 12, kernel_size=kernel_size, stride=1, device=DEVICE)
        self.bn1 = nn.GroupNorm(12, 12, device=DEVICE)
        # self.gnorm1 = nn.GroupNorm(6, 6, device=DEVICE)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(12, 24, kernel_size=kernel_size, stride=1, device=DEVICE)
        # self.gnorm2 = nn.GroupNorm(12, 24, device=DEVICE)
        self.bn2 = nn.GroupNorm(24, 24, device=DEVICE)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)


        # Decode layers 
        # ConvTranspose2d : https://cumulu-s.tistory.com/29
        self.tconv3 = nn.ConvTranspose2d(24, 12, kernel_size=2, stride=1, device=DEVICE)
        self.bn3 = nn.GroupNorm(12, 12, device=DEVICE)
        self.tconv4 = nn.ConvTranspose2d(12, 6, kernel_size=2, stride=1, device=DEVICE)

        self.fc = nn.Conv2d(6, 1, kernel_size=1, stride=1, padding=0, bias=True)

        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std


    def forward(self, x):
        x_embed = self.embed(x)
        pp(f"embed : {x_embed.size()}")
        x_conv1 = self.conv1(x_embed)
        pp(f"conv1 : {x_conv1.size()}")
        x_bn1 = self.bn1(x_conv1)
        pp(f"bn1 : {x_bn1.size()}")
        x_relu1 = self.relu1(x_bn1)
        pp(f"relu1 : {x_relu1.size()}")
        x_pool1 = self.pool1(x_relu1)
        pp(f"pool1 : {x_pool1.size()}")
        x_conv2 = self.conv2(x_pool1)
        pp(f"conv2 : {x_conv2.size()}")
        x_bn2 = self.bn2(x_conv2)
        pp(f"bn2 : {x_bn2.size()}")
        x_relu2 = self.relu2(x_bn2)
        pp(f"relu2 : {x_relu2.size()}")
        x_pool2 = self.pool2(x_relu2)
        pp(f"pool2 : {x_pool2.size()}")

        x_act = self.act(x_pool2)
        x_tconv3 = self.tconv3(x_act)
        pp(f"tconv3 : {x_tconv3.size()}")
        x_cat1 = torch.cat((x_tconv3, x_pool2), dim=1)
        x_bn3 = self.bn3(x_cat1)
        pp(f"bn3 : {x_bn3.size()}")

        x_tconv4 = self.tconv4(x_bn3)      
        pp(f"tconv4 : {x_tconv4.size()}")  
        x_cat2 = torch.cat((x_tconv4, x_pool2), dim=1)
        x_fc = self.fc(x_cat2)

        return x_fc



def unit_test():
    scoreNet = ScoreNet2D(1, 10, 2)
    print(scoreNet)


unit_test()




