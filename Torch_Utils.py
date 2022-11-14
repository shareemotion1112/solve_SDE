import torch
from Contant import DEVICE
import numpy as np

def convert_to_torch_tensor(arr):
    return torch.from_numpy(arr).float().to(DEVICE)

def convert_nparray_to_conv2d_input(input):
    input = input.cpu()
    input = np.expand_dims(input, axis=0)
    input = np.expand_dims(input, axis=0)
    input = convert_to_torch_tensor(input)
    input.to(DEVICE)
    return input


def get_magnitude(tensor):
    return torch.abs(tensor)

def get_trace(tensor):
    return torch.trace(tensor)

def get_optimizer(net, learning_rate):
    return torch.optim.Adam(net.parameters(), lr = learning_rate)