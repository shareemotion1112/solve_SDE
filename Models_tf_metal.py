import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python import keras
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Dense
from keras.losses import mse
from tqdm import trange, tqdm
import tensorflow_addons as tfa
import time
import math
import copy
# tensorflow에서는 마지막 차원이 channel
# input : [batch, in_height, in_width, in_channels] 형식. 28x28x1 형식의 손글씨 이미지.
# filter : [filter_height, filter_width, in_channels, out_channels] 형식. 3, 3, 1, 32의 w.


SIGMA = 10.
TIMESTEP = 1e-2
IS_TRAIN_MODEL = False
IS_SAVEFIG = True
batch_size = 32
predictor_steps = 20
corrector_steps = 20
numberOfFiles = 1024
output_dim = 1
epochs = 30

def get_rank(losses, training_loss):
    from copy import copy
    losses_c = copy(losses)
    losses_c.sort()
    rank = [ i for i, el in enumerate(losses_c) if training_loss > el]
    if rank == []:
        rank = [1]
    return len(rank)


def GaussianFourierProjection(x, embed_dim=256, scale=30):
    # tensorflow에서는 Weight, bias를 임의로 변경하는 것이 어려운 듯
    # keras 패키지의 미리 만들어진 Weight를 사용    
    
    W = tf.random.uniform((1, embed_dim // 2))
    x_proj = 2.0 * np.pi * scale * x
    x_proj = W * x_proj
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
    if name == "pc_sampler":
        eps = 1e-4
        num_steps = 1000
        time_steps = np.linspace(1., eps, num_steps)
        time_step = time_steps[1]
        step_size = time_steps[0] - time_steps[1]
        grad = scorenet(x, time_steps[1])
        grad_norm = tf.math.reduce_mean(tf.norm(tf.reshape(grad, (grad.shape[0], -1)), axis=-1))
        noise_norm = np.sqrt(np.prod(x.shape[1:])) # 400
        langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
        x = x + langevin_step_size * grad + tf.sqrt(2*langevin_step_size) * tf.random.uniform(x.shape)
        batch_time_step = tf.ones(batch_size) * time_step
        def sigma_func(t):
            return t ** 2
        g = sigma_func(batch_time_step)
        x_mean = x + (g**2) * scorenet(x, batch_time_step) * step_size
        x = x_mean + tf.sqrt(g**2 * step_size)[None, None, None, :] * tf.random.uniform(x.shape)

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

class myDense(keras.Model):
    def __init__(self, dim):
        super(myDense, self).__init__()
        self.dense = Dense(dim, dtype=tf.float32)
    def call(self, x):
        return self.dense(x)

class myConv2DTrans(keras.Model):
    def __init__(self, prev_x_dim, kernel_size, strides, padding="valid"):
        super(myConv2DTrans, self).__init__()
        self.convt = Conv2DTranspose(prev_x_dim, kernel_size=kernel_size, strides=strides, use_bias=False, padding=padding)
    def call(self, x):
        x = self.convt(x)
        return x

class myConv2D(keras.Model):
    def __init__(self, output_dim, kernel_size, strides, padding="same"):
        super(myConv2D, self).__init__()
        self.conv2d = Conv2D(output_dim, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False, name="conv2d")
    def call(self, x):
        return self.conv2d(x)

class myAct(keras.Model):
    def __init__(self):
        super(myAct, self).__init__()
        self.act = tf.sigmoid
    def call(self, x):
        x = x * self.act(x)
        return x

class myGN(keras.Model):
    def __init__(self, dim):
        super(myGN, self).__init__()
        self.gn = tfa.layers.GroupNormalization(dim)
    def call(self, x):
        return self.gn(x)

def ScoreNet2D(x, t, channels=[6, 12, 24, 48]):
    x_embed = GaussianFourierProjection(t)
    embed = myDense(256)(x_embed)

    #encoding path
    h1 = myConv2D(channels[0], 3, 1)(x)
    # test = myDense(channels[0])(embed)[..., None, None] # test.shape == [1, 6, 1, 1]
    h1 += myDense(channels[0])(embed)[None, None, ...]
    h1 = myGN(h1.shape[3])(h1)
    h1 = myAct()(h1)
    h2 = myConv2D(channels[1], 3, 2)(h1)
    h2 += myDense(channels[1])(embed)[None, None, ...]
    h2 = myGN(h1.shape[3])(h2)
    h2 = myAct()(h2)
    h3 = myConv2D(channels[2], 3, 2)(h2)
    h3 += myDense(channels[2])(embed)[None, None, ...]
    h3 = myGN(h2.shape[3])(h3)
    h3 = myAct()(h3)
    h4 = myConv2D(channels[3], 3, 2)(h3)
    h4 += myDense(channels[3])(embed)[None, None, ...]
    h4 = myGN(h3.shape[3])(h4)
    h4 = myAct()(h4)

    #decoding path
    h = myConv2DTrans(channels[2], 2, 2)(h4)
    h += myDense(channels[2])(embed)[None, None, ...]
    h = myGN(channels[2])(h)
    h = myAct()(h)
    h = myConv2DTrans(channels[1], 2, 2)(tf.concat([h, h3], axis=-1))
    h += myDense(channels[1])(embed)[None, None, ...]
    h = myGN(channels[1])(h)
    h = myConv2DTrans(channels[0], 2, 2)(tf.concat([h, h2], axis=-1))
    h += myDense(channels[0])(embed)[None, None, ...]
    h = myGN(channels[0])(h)
    h = myAct()(h)
    h = myConv2DTrans(1, 2, 1, "same")(tf.concat([h, h1], axis=-1))
    denominator = marginal_prob_std(t)
    out = h / denominator
    return out




import os
from PIL import Image
# from ImageHandle import get_img_dataloader
base_dir = "/Users/shareemotion/Projects/Solve_SDE/Data"
train_dir = os.path.join(base_dir,'train')
test_dir = os.path.join(base_dir,'test1')
file_names = os.listdir(train_dir)[:numberOfFiles]


class ImageDataset:
    def __init__(self, img_dir, file_names, batch_size=32, isResize=True):
        self.file_names = file_names
        self.img_dir = img_dir
        self.isResize = isResize
        self.batch_size=batch_size

    def normalize(self, img):
        return (img - np.min(img)) / (np.max(img) - np.min(img))

    def transform(self, img):
        min_width = 300
        min_height = 300
        img_cropped = img.crop(((img.size[0] - min_width)/2, \
                                (img.size[1] - min_height)/2, \
                                img.size[0] - (img.size[0] - min_width)/2, \
                                img.size[1] - (img.size[1] - min_height)/2))
        new_size = (400, 400)
        im = img_cropped.resize(new_size)
        return self.normalize(im)

    def __len__(self):
        return len(self.file_names) // self.batch_size

    def get_image_by_index(self, index):
        filename = self.file_names[index]
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path)
        if self.isResize:
            image = self.transform(image)
        label = filename.split('.')[0]
        im_arr = np.asarray(image)[:, :, 0]
        im = im_arr[None, :, :, None]
        return im, label

    def __getitem__(self, idx):
        res_im = None
        labels = []
        for i in range(idx * self.batch_size, (idx + 1) * self.batch_size, 1):
            im, label = self.get_image_by_index(i)
            labels.append(label)
            if res_im is None:
                res_im = im
            else:
                res_im = np.concatenate((res_im, im), axis=0)
        return res_im, labels

dataset = ImageDataset(train_dir, file_names, batch_size=batch_size)

if IS_TRAIN_MODEL is True:
    # train scoreNet
    train_loss = tf.keras.metrics.Mean()
    random_t = tf.random.uniform(shape=[])

    x = keras.Input((400, 400, 1))
    y = ScoreNet2D(x, random_t)
    scorenet = keras.Model(inputs=x, outputs=y)
    print(scorenet.summary())

    optimizer = Adam(learning_rate=1e-2)
    losses = []

    epoch = 0
    while True:
        num_items = 0
        for i in trange(len(dataset)):        
            x = dataset[i][0]
            num_items += x.shape[0]
            with tf.GradientTape() as tape:
                loss = loss_fn(scorenet, x, marginal_prob_std=marginal_prob_std)
                train_loss(loss)
            grads = tape.gradient(loss, scorenet.trainable_variables)
            optimizer.apply_gradients(zip(grads, scorenet.trainable_variables))
        current_loss = train_loss.result() / num_items
        losses.append(current_loss)
        print(f"{epoch} : {current_loss}")
        epoch += 1

        if get_rank(losses, current_loss) > 2:
            break;
        if epoch > epochs:
            break;

    pred = scorenet(x)
    plt.subplot(1, 2, 1)
    plt.imshow(x[0, :, :, 0])
    plt.title('original')
    plt.subplot(1, 2, 2)
    plt.imshow(pred[0, :, :, 0])
    plt.title('score')
    plt.subplots_adjust(hspace=0.5)
    plt.show(block=False)
    plt.pause(1)
    plt.close()


if IS_TRAIN_MODEL is False:
    # # model load
    scorenet = keras.models.load_model(os.getcwd() + '/tf_metal_model_1674783323')
else:
    # model save
    filename = 'tf_metal_model_' + str(math.ceil(time.time()))
    model_path = os.getcwd() + '/' + filename
    scorenet.save(model_path)





class VE_SDE:
    def __init__(self, n_batch, width, height, predictor_steps = 100, corrector_steps = 10, scoreNet = None):
        self.scoreNet = scoreNet
        self.predictor_steps = predictor_steps
        self.corrector_steps = corrector_steps        
        self.n_batch = n_batch
        self.width = width
        self.height = height
        self.epsilon = 1e-5    

    def diffusion_coef(self, t, sigma=SIGMA):
        return sigma ** t
    
    def run_pc_sampler(self, x, num_steps = 500, eps=1e-3, snr = 0.16):
        time_steps = np.linspace(1., eps, num_steps)
        step_size = time_steps[0] - time_steps[1]
        
        for i, time_step in enumerate(tqdm(time_steps)):
            batch_time_step = tf.ones(batch_size) * time_step

            # corrector step (Langevin MCMC)
            grad = self.scoreNet(x, time_step)
            grad_norm = tf.math.reduce_mean(tf.norm(tf.reshape(grad, (grad.shape[0], -1)), axis=-1))
            noise_norm = np.sqrt(np.prod(x.shape[1:])) # 400
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + tf.sqrt(2*langevin_step_size) * tf.random.uniform(x.shape)

            # predictor step (Euler-Maruyama)
            g = self.diffusion_coef(batch_time_step)
            x_mean = x + (g**2)[:, None, None, None] * scorenet(x, batch_time_step) * step_size
            a = x_mean
            b = tf.sqrt(g**2 * step_size)[:, None, None, None]
            c = tf.random.uniform(x.shape)
            x = tf.sqrt(g**2 * step_size)[:, None, None, None] * tf.random.uniform(x.shape)
            plt.imshow(x[0, :, :, :])
            plt.show(block=False)
            plt.pause(0.1)
            plt.close()
        # The last step does not include any noise !!!!!!
        return x_mean


ve_model = VE_SDE(batch_size, 400, 400, scoreNet=scorenet, \
                    predictor_steps = predictor_steps, corrector_steps=corrector_steps)
# for x, y in data_loader:
#     denoising_x = ve_model.run_denoising(x)
#     pp("denoising x : {denoising_x.shape}")
#     plot(x, scoreNet(x, 1), denoising_x)
    # predictor_x = ve_model.run_predictor_only(x)
    # plot(x, scoreNet(x, 1), predictor_x, denoising_x)

# # random matrix check : 아예 random한 데이터는 어려운 듯    
# x = torch.abs(torch.randn((1, 1, 400, 400)))
# for i in range(1000):
#     denoised_x = ve_model.run_denoising(x)        
#     x = denoised_x
#     plot(scoreNet(x, 1), denoised_x)
# import matplotlib.pyplot as plt
# plt.imshow(denoised_x[0, 0, :, :].cpu().detach().numpy(), cmap='gray')
# plt.show()

original_path = os.getcwd()
os.chdir(original_path + "/results")
# 데이터의 가운데를 지우고 테스트 

offset = 50
for x, y in dataset:
    x_cp = copy.copy(x)
    x_cp[:, (200-offset):(200+offset), (200-offset):(200+offset), :] = 0
    denoising_x = ve_model.run_pc_sampler(x_cp)

    image_name = str(math.ceil(time.time())) + '.png'

    plt.subplot(1, 2, 1)
    plt.imshow(x[0, :, :, :])
    plt.title('original')
    plt.subplot(1, 2, 2)
    plt.imshow(denoising_x[0, :, :, :])
    plt.title('denoised')
    plt.subplots_adjust(hspace=0.5)
    plt.show(block=False)    
    if IS_SAVEFIG is True:
        plt.savefig(image_name)
    plt.pause(1)
    plt.close()

os.chdir(original_path)