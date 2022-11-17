import torch
import torch.nn as nn
import torch.nn.functional as F
from Contant import DEVICE
import numpy as np
from cPrint import pp
from Torch_Utils import convert_to_torch_tensor


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
    x_proj = torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3] * 2)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp = x[i, j, :, :] * self.W * 2 * np.pi
            x_proj[i, j, :, :] = torch.cat([torch.sin(tmp), torch.cos(tmp)], dim=-1).to(DEVICE)
    return x_proj

def marginal_prob_std(t, sigma = 25):
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
channels, stride 설명 : 칼라갯수 또는 예측하고자 하는 클래스 수, filter가 이동하는 거리
http://taewan.kim/post/cnn/
https://justkode.kr/deep-learning/pytorch-cnn

Group Normalization :  batch의 크기가 극도로 작은 상황에서 batch normalization 대신 사용하면 좋은 normalization 기술
LN과 IN의 절충된 형태로 볼 수 있는데, 각 채널을 N개의 group으로 나누어 normalize 시켜주는 기술입니다.
https://blog.lunit.io/2018/04/12/group-normalization/

"""

class ScoreNet2D(nn.Module):
    def __init__(self, n_batch, n_channel, width, height, channels=[6, 12, 24, 48], kernel_size=2):
        super(ScoreNet2D, self).__init__()
        self.embed = GaussianFourierProjection()
        self.n_batch = n_batch
        self.n_channel = n_channel
        self.width = width
        self.height = height
        # Convolution, Dense(nn.Linear), Groupnormalization 순 : C -> D -> G


        self.conv1 = nn.Conv2d(n_channel, channels[0], kernel_size=kernel_size, stride=1, padding=1, device=DEVICE)
        self.gn1 = nn.GroupNorm(channels[0], channels[0], device=DEVICE)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=kernel_size, stride=1, padding=1, device=DEVICE)
        self.gn2 = nn.GroupNorm(channels[1], channels[1], device=DEVICE)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)


        # Decode layers 
        # ConvTranspose2d : https://cumulu-s.tistory.com/29
        self.tconv3 = nn.ConvTranspose2d(channels[1], channels[1], kernel_size=kernel_size, stride=2, padding=0, device=DEVICE)
        self.gn3 = nn.GroupNorm(channels[1], channels[2], device=DEVICE)
        self.tconv4 = nn.ConvTranspose2d(channels[2], channels[0], kernel_size=kernel_size, stride=2, padding=1, device=DEVICE)
        self.gn4 = nn.GroupNorm(channels[0], channels[1], device=DEVICE)

        self.conv_final = nn.Conv2d(channels[1], 1, kernel_size=1, stride=2, padding=0, device=DEVICE)

        self.dense1_input_size = 1
        # nn.Linear(n_channel * width * height, out_feature)
        self.dense1 = nn.Linear(self.dense1_input_size, n_batch * n_channel * width * height, device=DEVICE)


        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std


    def forward(self, x, t):
        if type(x) == np.ndarray:
            x = convert_to_torch_tensor(x)
        x_embed = self.embed(x)
        x_conv1 = self.conv1(x_embed)
        pp(f"x_conv1 : {x_conv1.size()}")
        x_gn1 = self.gn1(x_conv1)
        pp(f"x_gn1 : {x_gn1.size()}")
        x_relu1 = self.relu1(x_gn1)
        pp(f"x_relu1 : {x_relu1.size()}")
        x_pool1 = self.pool1(x_relu1)
        pp(f"x_pool1 : {x_pool1.size()}")
        x_conv2 = self.conv2(x_pool1)
        pp(f"x_conv2 : {x_conv2.size()}")
        x_gn2 = self.gn2(x_conv2)
        pp(f"x_gn2 : {x_gn2.size()}")
        x_relu2 = self.relu2(x_gn2)
        pp(f"x_relu2 : {x_relu2.size()}")
        x_pool2 = self.pool2(x_relu2)
        pp(f"x_pool2 : {x_pool2.size()}")

        x_act = self.act(x_pool2)
        
        # output_size 옵션으로 output의 dimension 조절
        x_tconv3 = self.tconv3(x_act, output_size=(x_conv2.shape[2], x_conv2.shape[3]))
        pp(f"x_conv2 : {x_conv2.size()}")
        pp(f"x_tconv3 : {x_tconv3.size()}")
        x_cat1 = torch.cat((x_conv2, x_tconv3), dim=1)
        pp(f"x_cat1 : {x_cat1.size()}")
        x_gn3 = self.gn3(x_cat1)
        pp(f"x_gn3 : {x_gn3.size()}")

        x_tconv4 = self.tconv4(x_gn3, output_size=(x_conv1.shape[2], x_conv1.shape[3]))      
        pp(f"x_tconv4 : {x_tconv4.size()}") 
        x_cat2 = torch.cat((x_conv1, x_tconv4), dim=1)
        pp(f"x_cat2 : {x_cat2.size()}")
        x_gn4 = self.gn4(x_cat2)
        pp(f"x_gn4 : {x_tconv4.size()}")
        x_conv_final = self.conv_final(x_gn4)
        pp(f"x_conv_final : {x_conv_final.size()}")
        denominator = marginal_prob_std(t)
        pp(f"t : {t}")
        pp(f"denominator : {denominator.size()}")

        x = x_conv_final / denominator
        self.dense1_input_size = x.shape[1] * x.shape[2] * x.shape[3]
        pp(f" the dense1_input_size : {self.dense1_input_size}")
        
        x.view(self.n_batch, -1)
        x = self.dense1(x)
        pp(f" x : {x.shape}")
        x = x.view(self.n_batch, -1)
        pp(f" x : {x.shape}")
        return x 



def unit_test():
    import matplotlib.pylab as plt
    x_height = 300
    y_height = 400
    n_batch = 10
    scoreNet = ScoreNet2D(n_batch=n_batch, n_channel=1, width=x_height, height=y_height)
    print(scoreNet)
    input = torch.randn(n_batch, 1, x_height, y_height).to(DEVICE)
    y = scoreNet(input, 1)

    yy = y[0, 0, :, :].detach().numpy()
    plt.subplot(2, 1, 1)
    plt.imshow(input[0, 0, :, :])
    plt.subplot(2, 1, 2)
    plt.imshow(yy)
    plt.show()


unit_test()




