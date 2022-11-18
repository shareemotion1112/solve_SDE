from PIL import Image
import os
import numpy as np
import copy
import matplotlib.pylab as plt
from functions import weight_function
from Models import ScoreNet2D
from Torch_Utils import *

# data download : https://www.kaggle.com/c/dogs-vs-cats/data


base_dir = "/Users/shareemotion/Projects/Solve_SDE/Data"

train_dir = os.path.join(base_dir,'train')
test_dir = os.path.join(base_dir,'test1')


file_names = os.listdir(train_dir)
img = Image.open(f"{train_dir}/{file_names[0]}")

im_arr = np.array(img)[:, :, 0]
im_arr.shape

row = 2
col = 4
t_limit = row * col
ratio = 100
new_im = copy.copy(im_arr)


isPlot = False

if isPlot is True:
    fig = plt.figure(figsize=(row * 4, col))

scoreNet = ScoreNet2D(1, 1)
opt = get_optimizer(scoreNet, 1e-3)

t = 0
while t < t_limit:
    t += 1
    ratio = weight_function(t, 2)
    z = np.random.rand(im_arr.shape[0], im_arr.shape[1]) * ratio

    new_im = new_im + z
    if isPlot is True:
        fig.add_subplot(row, col, t)
        plt.imshow(new_im)

    # input = convert_to_torch_tensor(new_im)
    input = convert_torchtensor_to_conv2d_input(new_im)
    out = scoreNet(input)
    scoreNet.zero_grad()
    
    loss = get_magnitude(out)**2 + get_trace(out)
    loss.backward()
    opt.step()

if isPlot is True:
    plt.show()

