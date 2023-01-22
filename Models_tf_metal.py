import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python import keras
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, ReLU, MaxPooling2D, Conv2DTranspose, Dense, BatchNormalization
from keras.losses import mse
from tqdm import trange
import tensorflow_addons as tfa


# tensorflow에서는 마지막 차원이 channel
# input : [batch, in_height, in_width, in_channels] 형식. 28x28x1 형식의 손글씨 이미지.
# filter : [filter_height, filter_width, in_channels, out_channels] 형식. 3, 3, 1, 32의 w.


SIGMA = 0.05
TIMESTEP = 0.01
tfa.layers.GroupNormalization()


def GaussianFourierProjection(x, scale=30):
    # tensorflow에서는 Weight, bias를 임의로 변경하는 것이 어려운 듯
    # keras 패키지의 미리 만들어진 Weight를 사용
    input_dim = x.shape[len(x.shape)-1]
    proj_kernel = Dense(input_dim, use_bias=False, trainable=False, kernel_initializer='identity', dtype=tf.float32)
    
    x_proj = 2.0 * np.pi * x
    x_proj = proj_kernel(x_proj) * scale

    x_proj_sin = tf.sin(x_proj)
    x_proj_cos = tf.cos(x_proj)

    output = tf.concat([x_proj_sin, x_proj_cos], axis=-1)
    return output

def marginal_prob_std(t, sigma = SIGMA):
    return tf.math.sqrt((sigma**(2 * t) - 1.) / 2. / tf.math.log(sigma))



def unit_test(name):
    if name == 'marginal_prob_std':
        # marginal probability test
        tt = [t/100 for t in range(100)]
        ttt = [ marginal_prob_std(t) for t in tt]
        plt.plot(tt, ttt);plt.show()


def loss_fn(model, x, marginal_prob_std, eps=1e-5): # ------------------ random t 를 사용??
    random_t = tf.random.uniform(shape=[]) * (1. - eps) + eps    
    z = tf.random.uniform(shape=x.shape, minval=0, maxval=255, dtype=tf.int32)
    z = tf.cast(z, dtype=tf.float32)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std
    score = model(perturbed_x, random_t)
    # print(f"score dimension : {score.shape}")
    sum = tf.reduce_sum((score * std + z)**2)

    loss = tf.reduce_mean(sum)
    return loss

class DownSample(keras.Model):
    def __init__(self, output_dim):
        super(DownSample, self).__init__()
        self.output_dim = output_dim   
        self.conv2d = Conv2D(self.output_dim, kernel_size=3, strides=1, padding="same", name="conv2d")
        self.gn = tfa.layers.GroupNormalization(self.output_dim)
        self.relu = ReLU()
        self.maxpool = MaxPooling2D((2, 2))

    def call(self, x, training=False):
        x = self.conv2d(x)
        x = self.gn(x, training=training)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class UpSample(keras.Model):
    def __init__(self):
        super(UpSample, self).__init__()
        self.conv2d = Conv2D(x.shape[3], kernel_size=3, strides=1, padding="same", name="Conv2d_upsample")
        self.conv2dT = Conv2DTranspose(x.shape[3], kernel_size=2, strides=2, padding="valid", name="conv2dTranspose_upsample")
        self.gn = tfa.layers.GroupNormalization(x.shape[3])
        self.relu = ReLU()

    def call(self, prev_x, x, training=False):
        x = self.conv2d(x)
        x = tf.concat([prev_x, x], axis=3)
        x = self.conv2dT(x)
        x = self.gn(x, training=training)
        return x

class myConv2DTrans(keras.Model):
    def __init__(self, prev_x_dim):
        super(myConv2DTrans, self).__init__()
        self.convt = Conv2DTranspose(prev_x_dim, kernel_size=2, strides=2, padding="valid")
    def call(self, x):
        x = self.convt(x)
        return x

class myConv2D(keras.Model):
    def __init__(self):
        super(myConv2D, self).__init__()
        self.conv = Conv2D(1, kernel_size=3, strides=1, padding="same")
    def call(self, x):
        x = self.conv(x)
        return x

def ScoreNet2D(x, t, channels=[6, 12, 24, 48]):
    x_embed = GaussianFourierProjection(x)
    x_down1 = DownSample(channels[0])(x_embed)
    x_down2 = DownSample(channels[1])(x_down1)
    x_btm = DownSample(channels[2])(x_down2)    
    x_btm2 = myConv2DTrans(x_down2.shape[3])(x_btm)
    x_up1 = UpSample()(x_down2, x_btm2)
    x_up2 = UpSample()(x_down1, x_up1)
    x = myConv2D()(x_up2)
    x_act = tf.math.sigmoid(x)
    denominator = marginal_prob_std(t)
    return x_act / denominator


class VE_SDE:
    def __init__(self, n_batch, width, height, predictor_steps = 100, corrector_steps = 10, scoreNet = None):
        self.scoreNet = scoreNet
        self.predictor_steps = predictor_steps
        self.corrector_steps = corrector_steps
        self.drift_coef = self.drift_func
        self.diffusion_coef = 0
        self.n_batch = n_batch
        self.width = width
        self.height = height
        self.epsilon = 1e-5

    def sigma_func(self, t):
        return t ** 2

    def drift_func(self, t):
        return tf.math.sqrt(2 * t) * tf.random.uniform(shape=[])

    def predictor(self, x, i):        
        t = i * TIMESTEP
        sigma_diff = (self.sigma_func(t + 1)**2 - self.sigma_func(t)**2)        
        x_i_prime = x + sigma_diff * self.scoreNet(x, t)
        z = tf.random.uniform(shape=[])
        x = x_i_prime + tf.math.sqrt(sigma_diff) * z            
        return x
    def corrector(self, x, j):
        t = (j + 1) * TIMESTEP
        z = tf.random.uniform(shape=[])

        x = x + self.epsilon * self.scoreNet(x, t) + tf.math.sqrt(tf.math.abs(2 * self.epsilon)) * z        
        return x

    def run_denoising(self, x):
        for i in trange(self.predictor_steps -1, 0, -1):
            x = self.predictor(x, i)
            for j in range(0, self.corrector_steps, 1):
                x = self.corrector(x, j)
        return x

import os
from PIL import Image
# from ImageHandle import get_img_dataloader
base_dir = "/Users/shareemotion/Projects/Solve_SDE/Data"
batch_size = 1
predictor_steps = 10 # 너무 노이즈를 많이 넣어도 학습이 안될 듯
corrector_steps = 50
train_dir = os.path.join(base_dir,'train')
test_dir = os.path.join(base_dir,'test1')
file_names = os.listdir(train_dir)[:100]

# file_path = os.path.join(train_dir, file_names[0])
# img = Image.open(file_path)
# plt.imshow(im); plt.show()


class ImageDataset:
    def __init__(self, img_dir, file_names, isResize=True):
        self.file_names = file_names
        self.img_dir = img_dir
        self.isResize = isResize

    def transform(self, img):
        min_width = 300
        min_height = 300
        img_cropped = img.crop(((img.size[0] - min_width)/2, (img.size[1] - min_height)/2, img.size[0] - (img.size[0] - min_width)/2, img.size[1] - (img.size[1] - min_height)/2))
        new_size = (400, 400)
        im = img_cropped.resize(new_size)
        return im

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        filename = self.file_names[idx]
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path)
        if self.isResize:
            image = self.transform(image)
        label = filename.split('.')[0]
        im_arr = np.asarray(image)[:, :, 0]
        im = im_arr[None, :, :, None]
        return im, label

dataset = ImageDataset(train_dir, file_names)



output_dim = 1
epochs = 200
train_loss = tf.keras.metrics.Mean()
random_t = tf.random.uniform(shape=[])

x = keras.Input((400, 400, 1))
y = ScoreNet2D(x, random_t)
model = keras.Model(inputs=x, outputs=y)
print(model.summary())
# print(model.trainable_variables)
# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.enable_eager_execution()

optimizer = Adam(learning_rate=1e-1)
losses = []
for epoch in range(epochs):
    for x, label in dataset:        
        with tf.GradientTape() as tape:
            loss = loss_fn(model, x, marginal_prob_std=marginal_prob_std)
            train_loss(loss)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    losses.append(train_loss.result())
    print(f"{epoch} : {train_loss.result()}")

pred = model(x)

plt.subplot(1, 2, 1)
plt.imshow(x[0, :, :, :])
plt.title('original')
plt.subplot(1, 2, 2)
plt.imshow(pred[0, :, :, :])
plt.title('score')
plt.subplots_adjust(hspace=0.5)
plt.show()

print(f"max : {np.max(pred)}, min : {np.min(pred)}")