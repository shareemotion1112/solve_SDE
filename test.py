
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    xx = x[:, None]
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi # x_proj.shape == [1, 128], x.shape == [1]
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


t = torch.randn(1); print(t.shape)
tt = GaussianFourierProjection(256)(t); print(tt.shape)
ttt = Dense(256, 32)(tt); print(ttt.shape)

x = torch.randn((400, 400, 3))
print(x.shape)
x1 = Dense(3, 16)(x); print(x1.shape)