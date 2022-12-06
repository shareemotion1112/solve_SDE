from PIL import Image
import os
import numpy as np
import copy
from Models import ScoreNet2D, loss_fn, marginal_prob_std
from Utils import *
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
from torchvision import transforms
from torchvision.io import read_image
from ImageHandle import ImageDataset

# data download : https://www.kaggle.com/c/dogs-vs-cats/data


base_dir = "/Users/shareemotion/Projects/Solve_SDE/Data"
isPlot = False
batch_size = 3
n_channel = 1


train_dir = os.path.join(base_dir,'train')
test_dir = os.path.join(base_dir,'test1')


file_names = os.listdir(train_dir)
img = Image.open(f"{train_dir}/{file_names[0]}")
im_arr = np.array(img)[:, :, 0]
im_arr.shape


transform = torch.nn.Sequential(transforms.CenterCrop((300, 300)), transforms.Resize((400, 400)))
train_dataset = ImageDataset(train_dir, file_names[:100], transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)


scoreNet = torch.nn.DataParallel(ScoreNet2D(batch_size, n_channel, im_arr.shape[0], im_arr.shape[1]))
scoreNet = scoreNet.to(DEVICE)
opt = get_optimizer(scoreNet, 1e-3)

n_epochs = 50

tqdm_epoch = trange(n_epochs)
for epoch in tqdm_epoch:
    
    avg_loss = 0.
    num_items = 0

    for x, y in train_dataloader:
        x = x.to(DEVICE)
        # print(f'x in train_dataloader : {x.shape}')
        loss = loss_fn(scoreNet, x, marginal_prob_std = marginal_prob_std)
        opt.zero_grad()
        loss.backward()
        opt.step()

        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
