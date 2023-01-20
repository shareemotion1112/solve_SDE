import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python import keras
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, LayerNormalization, ReLU, MaxPooling2D, Conv2DTranspose, Dense
from keras.losses import mse
from tqdm import trange

# tensorflow에서는 마지막 차원이 channel
# input : [batch, in_height, in_width, in_channels] 형식. 28x28x1 형식의 손글씨 이미지.
# filter : [filter_height, filter_width, in_channels, out_channels] 형식. 3, 3, 1, 32의 w.


SIGMA = 0.05
TIMESTEP = 0.01



def GaussianFourierProjection(x, scale=30):
    # tensorflow에서는 Weight, bias를 임의로 변경하는 것이 어려운 듯
    # keras 패키지의 미리 만들어진 Weight를 사용
    input_dim = x.shape[len(x.shape)-1]
    proj_kernel = Dense(input_dim, use_bias=False, trainable=False, kernel_initializer='identity', dtype=tf.float32)
    
    x_proj = 2.0 * np.pi * x
    x_proj = proj_kernel(x_proj)

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
    random_t = tf.random.uniform(shape=[x.shape[0]]) * (1. - eps) + eps    
    z = tf.random.uniform(shape=x.shape, minval=0, maxval=255, dtype=tf.int32)
    z = tf.cast(z, dtype=tf.float32)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std
    score = model(perturbed_x, random_t)

    sum = tf.reduce_sum((score * std + z)**2)

    loss = tf.reduce_mean(sum)
    return loss

# from keras.models import Sequential
# from keras.layers import Dense
# X = tf.keras.layers.Input(shape=[28, 28, 1])
# def get_model(n_x, n_h1, n_h2):
#     model = Sequential()
#     model.add(Dense(n_h1, input_dim=n_x, activation='relu'))
#     model.add(Dense(n_h2, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(4, activation='softmax'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     print(model.summary())
#     return model


def DownSample(x, output_dim): 
    x_conv = Conv2D(output_dim, kernel_size=3, strides=1, padding="same")(x)
    x_ln = LayerNormalization()(x_conv)
    x_relu = ReLU()(x_ln)
    x_mp = MaxPooling2D((2, 2))(x_relu)
    return x_mp

def UpSample(prev_x, x):
    x = Conv2D(x.shape[3], kernel_size=3, strides=1, padding="same")(x)
    x = tf.concat([prev_x, x], axis=3)
    x = Conv2DTranspose(x.shape[3], kernel_size=2, strides=2, padding="valid")(x)
    x = LayerNormalization()(x)
    return x


def ScoreNet2D(x, t, channels=[6, 12, 24, 48]):
    x_embed = GaussianFourierProjection(x)
    x_down1 = DownSample(x_embed, channels[0])
    x_down2 = DownSample(x_down1, channels[1])
    x_btm = DownSample(x_down2, channels[2])
    x_btm2 = Conv2DTranspose(x_down2.shape[3], kernel_size=2, strides=2, padding="valid")(x_btm)
    x_up1 = UpSample(x_down2, x_btm2)
    x_up2 = UpSample(x_down1, x_up1)
    x = Conv2D(1, kernel_size=3, strides=1, padding="same")(x_up2)
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


def train_scoreNet(data_loader, batch_size, width, height):
    scoreNet = ScoreNet2D(batch_size, 1, width, height)
    optim = Adam(learning_rate=1e-4)

    epochs = 10
    for x, y in data_loader:
        print('|', end="")
        for i in range(epochs):
            scoreNet_loss = loss_fn(scoreNet, x, marginal_prob_std = marginal_prob_std)

            
    return scoreNet

# import os
# from ImageHandle import get_img_dataloader
base_dir = "/Users/shareemotion/Projects/Solve_SDE/Data"
batch_size = 1
predictor_steps = 10 # 너무 노이즈를 많이 넣어도 학습이 안될 듯
corrector_steps = 50
# train_dir = os.path.join(base_dir,'train')
# test_dir = os.path.join(base_dir,'test1')
# file_names = os.listdir(train_dir)[:100]
# scoreNet = train_scoreNet(data_loader, batch_size, 400, 400)


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images.shape

# x_down = DownSample(x, 1)

x = tf.random.uniform((1, 400, 400, 1)); output_dim = 1
inputs = keras.Input(shape=(x.shape[1], x.shape[2], x.shape[3]), name='input')
output_dim = 1; print(x.shape)

train_loss = tf.keras.metrics.Mean()
optimizer = Adam(learning_rate=1e-4)
with tf.GradientTape() as tape:
    random_t = tf.random.uniform(shape=[])
    outputs = ScoreNet2D(inputs, random_t)
    model = keras.Model(inputs, outputs)
    model.summary()
    loss = loss_fn(model, x, marginal_prob_std=marginal_prob_std)
    train_loss(loss)
grads = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))

print(train_loss.result())

# train_acc = tf.keras.metrics.SparseCategoricalAccuracy()