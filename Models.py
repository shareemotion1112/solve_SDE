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
        super(ScoreNet2D, self).__init__()
        self.embed_dim = embed_dim
        self.n_channel = n_channel
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim), \
         nn.Linear(embed_dim, embed_dim))
        self.conv1 = nn.Conv2d(n_batch, 1, kernel_size=kernel_size, stride=1, device=DEVICE)
        self.gnorm1 = nn.GroupNorm(32, num_channels=n_channel, device=DEVICE)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, device=DEVICE)
        self.gnorm2 = nn.GroupNorm(32, num_channels=n_channel, device=DEVICE)

        self.fc1 = nn.Linear(94, 32, device=DEVICE)
        self.fc2 = nn.Linear(32, 16, device=DEVICE)

        # Decode layers 
        # ConvTranspose2d : https://cumulu-s.tistory.com/29
        self.tconv3 = nn.ConvTranspose2d(16, 32, kernel_size=kernel_size, stride=1, device=DEVICE)
        self.fc3 = nn.Linear(embed_dim, 32)
        self.tconv4 = nn.ConvTranspose2d(64, 1, kernel_size=kernel_size, stride=1, device=DEVICE)

        self.act = lambda x: x * torch.sigmoid(x)


    def forward(self, x):
        x = self.gnorm1(self.conv1(x))
        x = self.gnorm2(self.conv2(x))
        x = nn.BatchNorm1d(self.fc1(x), device=DEVICE)
        x = self.act(self.fc2(x))
        x = self.tconv3(x)
        x = self.fc3(x)
        x = self.tconv4(x)        
        return x



def unit_test():
    scoreNet = ScoreNet2D(1, 10, 2)
    print(scoreNet)