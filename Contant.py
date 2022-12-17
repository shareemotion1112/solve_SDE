import torch
import platform



# if platform.platform()[:5] == 'macOS':
#     DEVICE = torch.device('mps')
# else:
    # DEVICE = torch.device('cuda:0')
# DEVICE = torch.device('cpu')
DEVICE = torch.device('cuda:0')
# DEVICE = torch.device('mps')