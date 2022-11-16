import torch
import platform


# DEVICE = torch.device('mps')
# if platform.platform()[:5] == 'macOS':
#     DEVICE = torch.device('mps')
# else:
#     DEVICE = torch.device('cuda:0')
DEVICE = torch.device('cpu')