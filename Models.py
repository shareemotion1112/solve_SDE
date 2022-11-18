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
    x_proj = torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp = x[i, j, :, :] * self.W * 2 * np.pi
            x_proj[i, j, :, :] = torch.sin(tmp) + torch.cos(tmp) # concat을 하던데 꼭 그럴 필요가 잇나싶음 ..... 체크 필요!!
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

    pp(f"score : {score.shape}, std : {std.shape}, z : {z.shape}"); pp(f"{(score * std[:, None, None, None] + z)**2}")
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

ConvTranspose2d : https://cumulu-s.tistory.com/29

"""


class DownSample(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1)
        self.gn = nn.GroupNorm(output_dim, output_dim)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
    def forward(self, x):
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
        self.conv = None
        self.gn = None

    def tconv(self, n_ch):
        return nn.ConvTranspose2d(n_ch, n_ch, kernel_size=2, stride=2, padding=0)
        
    def forward(self, previous_x, x):
        self.input_dim = x.shape[1]
        self.conv = nn.Conv2d(self.input_dim, self.input_dim // 2, kernel_size=3, stride=1, padding=1)
        # self.gn = nn.GroupNorm(self.input_dim // 2, self.input_dim // 2)        

        x = self.conv(x)
        pp(f"x : {x.shape}")
        x = torch.cat([previous_x, x], dim=1)
        pp(f"shape check : {previous_x.shape} vs {x.shape}")
        pp(f"x : {x.shape}")
        self.input_dim = x.shape[1]
        x = self.tconv(x.shape[1])(x, output_size=(previous_x.shape[2] * 2, previous_x.shape[3] * 2))
        x = nn.GroupNorm(x.shape[1], x.shape[1])(x)
        
        return x    

class ScoreNet2D(nn.Module):
    def __init__(self, n_batch, n_channel, width, height, channels=[6, 12, 24, 48]):
        super(ScoreNet2D, self).__init__()
        
        self.n_batch = n_batch
        self.n_channel = n_channel
        self.width = width
        self.height = height

        
        self.embed = GaussianFourierProjection()
        
        self.down1 = DownSample(n_channel, channels[0])
        self.down2 = DownSample(channels[0], channels[1])

        self.bottom = DownSample(channels[1], channels[2])

        self.up1 = UpSample()
        self.up2 = UpSample()



        self.dense1_input_size = 1        
        self.dense1 = nn.Linear(self.dense1_input_size, n_batch * n_channel * width * height)


        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std


    def forward(self, x, t): # t는 배치숫자와 동일해야함
        if type(x) == np.ndarray:
            x = convert_to_torch_tensor(x)
        x_embed = self.embed(x); pp(f"x_embed : {x_embed.shape}")
        x_down1 = self.down1(x_embed); pp(f"x_down1 : {x_down1.size()}")
        x_down2 = self.down2(x_down1); pp(f"x_down2 : {x_down2.size()}")

        x_bottom = self.bottom(x_down2); pp(f"x_bottom : {x_bottom.size()}")
        x_bottom = nn.ConvTranspose2d(x_bottom.shape[1], x_bottom.shape[1], kernel_size=2, stride=2, padding=0)(x_bottom)
        pp(f"x_bottom_conv : {x_bottom.size()}")

        x_up1 = self.up1(x_down2, x_bottom); pp(f"x_up1 : {x_up1.shape}")
        x_up2 = self.up2(x_down1, x_up1); pp(f"x_up2 : {x_up2.shape}")
        x_act = self.act(x_up2); pp(f"x_act : {x_act.shape}")

        denominator = marginal_prob_std(t); pp(f"t : {t}"); pp(f"denominator : {denominator.size()}")
        
        x = x_act / denominator[:, None, None, None]
        pp(f"final x : {x.shape}")
        
        return x 



def unit_test():
    import matplotlib.pylab as plt
    x_height = 400
    y_height = 400
    n_batch = 10
    scoreNet = ScoreNet2D(n_batch=n_batch, n_channel=1, width=x_height, height=y_height)
    scoreNet.to(DEVICE)
    print(scoreNet)
    input = torch.randn(n_batch, 1, x_height, y_height).to(DEVICE)

    
    loss = loss_fn(scoreNet, input, marginal_prob_std = marginal_prob_std)
    optimizer = torch.optim.Adam(scoreNet.parameters(), lr = 1e-4)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    eps = 1e-5
    t = torch.rand(input.shape[0], device=DEVICE) * (1. - eps) + eps  # 왜 random T를 사용? int 스텝을 넣는게 아니네??
    y = scoreNet(input, t)
    yy = y[0, 0, :, :].detach().numpy()
    plt.subplot(2, 1, 1)
    plt.imshow(input[0, 0, :, :])
    plt.subplot(2, 1, 2)
    plt.imshow(yy)
    plt.show()


unit_test()




