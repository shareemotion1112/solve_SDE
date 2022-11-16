import torch
import torch.nn as nn
import torch.nn.functional as F
from Contant import DEVICE
import numpy as np

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
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

class ScoreNet1D(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ScoreNet1D, self).__init__()
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
    def __init__(self, n_batch, n_channel = 1, kernel_size=2, embed_dim=256):
        """
        n_channel : RGB 개수
        """
        super(ScoreNet2D, self).__init__()
        self.embed_dim = embed_dim
        self.n_channel = n_channel
        self.GFP = GaussianFourierProjection(embed_dim=embed_dim)
        self.embed = nn.Linear(embed_dim, embed_dim)
        self.conv1 = nn.Conv2d(n_batch, 6, kernel_size=kernel_size, stride=1, device=DEVICE)
        self.gnorm1 = nn.GroupNorm(6, num_channels=36, device=DEVICE)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=kernel_size, stride=1, device=DEVICE)
        self.gnorm2 = nn.GroupNorm(12, num_channels=36, device=DEVICE)

        self.fc1 = nn.Linear(94, 32, device=DEVICE)
        self.fc2 = nn.Linear(32, 16, device=DEVICE)

        # Decode layers 
        # ConvTranspose2d : https://cumulu-s.tistory.com/29
        self.tconv3 = nn.ConvTranspose2d(16, 32, kernel_size=kernel_size, stride=1, device=DEVICE)
        self.fc3 = nn.Linear(embed_dim, 32)
        self.tconv4 = nn.ConvTranspose2d(64, 1, kernel_size=kernel_size, stride=1, device=DEVICE)

        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std


    def forward(self, x, t):
        x_gfp = self.GFP(x)
        x_embed = self.embed(x_gfp)
        x_conv1 = self.conv1(x)
        x_gnorm1 = self.gnorm1(x_conv1)
        x_fc1 = self.fc1(x_gnorm1)
        x_conv2 = self.conv2(x_fc1)
        x_gnorm2 = self.gnorm2(x_conv2)
        x_fc2 = self.fc2(x_gnorm2)

        x_tconv3 = self.tconv3(x_fc2)
        x_cat1 = torch.cat((x_tconv3, ))
        x_fc3 = self.fc3(x_tconv3)
        x_tconv4 = self.tconv4(x_fc3)


        x = x / self.marginal_prob_std(t)[:, None, None, None]
        return x



def unit_test():
    scoreNet = ScoreNet2D(1, 10, 2)
    print(scoreNet)


unit_test()




