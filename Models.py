
import torch.nn as nn
import torch.nn.functional as F
from Contant import DEVICE




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

class ScoreNet2D(nn.Module):
    def __init__(self, in_channel, output_dim, kernel_size=2):
        super(ScoreNet2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 12, kernel_size=kernel_size, device=DEVICE)
        self.conv2 = nn.Conv2d(12, 6, kernel_size=kernel_size, device=DEVICE)

        self.fc1 = nn.Linear(94, 32, device=DEVICE)
        self.fc2 = nn.Linear(16, output_dim, device=DEVICE)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), (2, 2))
        x = F.max_pool2d(self.conv2(x), (2, 2))
        x = nn.BatchNorm1d(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x



def unit_test():
    scoreNet = ScoreNet2D(1, 10, 2)
    print(scoreNet)