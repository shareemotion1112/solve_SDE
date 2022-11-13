from PIL import Image
import os
# import cv2
import numpy as np
import copy
import matplotlib.pylab as plt
from functions import weight_function


# data download : https://www.kaggle.com/c/dogs-vs-cats/data


base_dir = "/Users/shareemotion/Projects/Solve_SDE/Data"

train_dir = os.path.join(base_dir,'train')
test_dir = os.path.join(base_dir,'test1')


file_names = os.listdir(train_dir)

# img =   cv2.imread(file_names[0])
# cv2.imshow('aa' , img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img = Image.open(f"{train_dir}/{file_names[0]}")
# img.show()
# img.shape

im = np.array(img)[:, :, 0]
im.shape

row = 2
col = 4
t_limit = row * col
ratio = 100
new_im = copy.copy(im)
t = 0

fig = plt.figure(figsize=(row*4, col))

while t < t_limit:
    t += 1
    ratio = weight_function(t, 2)
    z = np.random.rand(im.shape[0], im.shape[1]) * ratio

    new_im = new_im + z
    fig.add_subplot(row, col, t)
    plt.imshow(new_im)

plt.show()
