import torch
import platform



if platform.platform()[:5] == 'macOS':
    # DEVICE = torch.device('mps')
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cuda:0')
