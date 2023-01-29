import torch
from Contant import DEVICE
import numpy as np

def convert_to_torch_tensor(arr):
    return torch.from_numpy(arr).float().to(DEVICE)

def convert_torchtensor_to_conv2d_input(input):
    input = input.cpu()
    input = np.expand_dims(input, axis=0)
    input = np.expand_dims(input, axis=0)
    input = convert_to_torch_tensor(input)
    input.to(DEVICE)
    return input

def get_magnitude(tensor):
    return torch.inner(torch.transpose(tensor, 0, 1), tensor)

def get_trace(tensor):
    return torch.trace(tensor)

def get_optimizer(net, learning_rate=1e-4):
    return torch.optim.Adam(net.parameters(), lr = learning_rate)


def plot(*args, name=None):
    import matplotlib.pyplot as plt
    num_plots = len(args)
    for i, arg in enumerate(args):
        plt.subplot(1, num_plots, i + 1)
        plt.imshow(arg[0, 0, :, :].cpu().detach().numpy(), cmap='gray')
        if name is not None:
            plt.title(name[i])
    plt.subplots_adjust(hspace=0.5)
    plt.show(block=False)
    plt.pause(1)
    plt.close()


def plot_imgs(x):
    n_images = x.shape[0]
    n_row = int(np.ceil(np.sqrt(n_images)))
    n_col = int(n_row + 1)
    for i in range(1, n_images):
        plt.subplot(n_row, n_col, i)
        plt.imshow(x[i, :, :, :])        
        plt.axis('off')    
    plt.subplots_adjust(hspace=0)
    plt.show(block=False)


def get_rank(losses, training_loss):
    from copy import copy
    losses_c = copy(losses)
    losses_c.sort()
    rank = [ i for i, el in enumerate(losses_c) if training_loss > el]
    if rank == []:
        rank = [1]

    return len(rank)

