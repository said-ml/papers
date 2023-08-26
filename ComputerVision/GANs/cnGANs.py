​
###########################--------------- suppressing  the necessary warnings(optionl) -------------#######################
​
import sys
if not sys.warnoptions:
    import warnings
    warnings.filterwarnings('ignore')
​
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# i use this to hide some warning (i'am IDE pycharm's user), so in most cases this optional
​
#############################------------------# import  dependencies-----------###################################
​
import tensorflow as tf
# leveraging tensorflow
import numpy as np
# for manipulating linar algebra's calculs
import matplotlib.pyplot as plt
# for data visualization
from tensorflow import keras
# keras is a high level for deep learning
from keras.datasets import mnist
from keras.activations import sigmoid
from keras.losses import BinaryCrossentropy
from keras.utils.vis_utils import plot_model
# ploting architecture of the model
from tqdm import tqdm
# for training progress
from keras import layers
​
###########################----------- setup the enveriment( make use GPU when it's available)-------------##################################
'''
GPU_is_available=tf.test.is_built_with_cuda()
GPU_list_devices=tf.config.list_physical_devices()
if GPU_is_available:
    # use the first device
    tf.config.set_visible_devices(GPU_list_devices[0], 'GPU')
# the commands above are useless if you workin in the cloud(google.colab for example)
'''   
##########################------------preprocessing data----------###########################
​
np.random.seed(0)
# to make the ouput of the code unique
​
(X_train, _), (X_test, _) = mnist.load_data()
# loading mnist images and showing some samples
def plot_save_images(plot_real_images=True, epoch=None, n_samples=16):
    for i in range(n_samples):
        # define subplot
        plt.subplot(4, 4, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(X_train[i], cmap='gray_r')
    plt.show()
    plt.savefig('real handsdigits.png')
plot_save_images()
​
def preprocessing_data(batch_size=32, buffer_size=1, drop_remainder=True):
    (X_train, _), (X_test, _) = mnist.load_data()
    assert X_train.shape==(60000, 28,28)
    # 60000 is the number of the traing images and each images has 28x28 pixels
    assert X_test.shape == (10000, 28, 28)
    # 10000 is the number of the test images
    X_train=X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    # reshaping normalizing the images
    X_train = (X_train-127.5)/127.5
    X_train=tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000)
    # you can use random shuffle in order to mix up the data through tf.random.shuffle(X_train) but batch function doesn't available in this case
    X_train=X_train.batch(batch_size=batch_size, drop_remainder=drop_remainder).prefetch(buffer_size=buffer_size)
    return X_train
    
​
#preprocessing_data(train=True, batch_size=32, buffer_size=1, drop_remainder=True)
​
#######################-------gans building:1==>builduing descriminator-----------#########################
​
def discriminator():
    model = tf.keras.Sequential(name='discriminator')
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
​
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
​
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
​
    return model
descriminator=discriminator()
#descriminator.summary()
#plot_model(descriminator, to_file='descriminator.png', show_shapes=True, show_layer_names=True)
​
# define the loss of descriminator
entropy=BinaryCrossentropy()
def d_loss(real, fake):
        real_loss = entropy(tf.ones_like(real), real)
        fake_loss = entropy(tf.zeros_like(fake), fake)
        total_loss = real_loss + fake_loss
        return total_loss
​
###################-------------------gans building:2)==> building the generator---------###########################
​
def generator():
    model = tf.keras.Sequential(name='generator')
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
​
    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size
​
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
​
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
​
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)
​
    return model
​
generator=generator()
​
# ploting the architecture of the generator
#generator.summary()
#plot_model(generator, to_file='generator.png', show_shapes=True, show_layer_names=True)
# defining the loss of the generator g_loss
def g_loss(fake):
    return entropy(tf.ones_like(fake), fake)
​
####################---------set up the hyperparameters----------###############################
​
batch_size=16
generator_opt=keras.optimizers.Adam(learning_rate=0.0001)
descriminator_opt=keras.optimizers.Adam(learning_rate=0.0001)
latent_dim=100
# we draw latent vector from the latent space (using gaussian distribution)
latent_vector=tf.random.normal((batch_size, latent_dim))
epochs=250
​
# there a confrontation between generator and descriminator , the minimizing of g_loss can lead to explosion of d_loss and vice versa, gans.fit(...)
# we must define a custom loop
​
# firstly we create method that perform one single training step
@tf.function
# this annotation for better performance(it compile the function)
def batch_train_step(batch_size, images, latent_dim):
    # we will use tf.Gradient as a tool for automatic differentiation (autodiff), instead of optimizer=... as argument in fit(.) method
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        # feed noise to the generator to produces images
        generated_images=generator(latent_vector, training=True)
        #generator.trainable=True
        # we release the weights of descriminator to train him to distinguish between real and fake images
        real=descriminator(images, training=True)
        fake=descriminator(generated_images, training=True)
        # we calcul respectively the losses g_loss and d_loss
        generator_loss=g_loss(fake)
        descriminator_loss=d_loss(real, fake)
        # we compute the gradient of generator_loss (g_gradient) with respect to the weights of the generator
        g_gradients=g_tape.gradient(generator_loss, generator.trainable_variables)
        # we compute the gradient of descriminator_loss (g_gradient) with respect to the weights of the descriminator
        d_gradients=d_tape.gradient(descriminator_loss, descriminator.trainable_variables)
        # applying the gradients in order to update the weights of generator
        g_arguments=zip(g_gradients, generator.trainable_variables)
        generator_opt.apply_gradients(g_arguments)
        # applying the gradients in order to update the weights of  descriminator
        d_arguments = zip(d_gradients, descriminator.trainable_variables)
        descriminator_opt.apply_gradients(d_arguments)
       
# now let's iterate over all EPOCHS:
​
dataset =preprocessing_data()
​
def fit(dataset, epochs, latent_dim):
    for epoch in tqdm(range(epochs)):
        # tqdm for the training progress
        for batch_images in dataset:
                # applying batch_train_step
                batch_train_step(batch_size, batch_images, latent_dim)
                generated_images=generator(latent_vector)
        # iterate over generate images
        if epoch//20==0:
                for i in range(generated_images.shape[0]):
                      plt.subplot(4, 4, i +1)
                      image=generated_images[i]
                      # rescale the image
                      image=image*127.5+127.5
                      # visualize it
                      plt.imshow(image, cmap='gray')
                      plt.axis('off')
                # save generated images with their corresponding epoch to maintain control of the training progress
                plt.savefig(f'{epoch}.png')
                plt.show()
​
# Batch and shuffle the data
dataset=preprocessing_data()
epochs=200
latent_dim=100
​
​
if __name__ == '__main__':
​
    # training GANs
    fit(dataset, epochs, latent_dim )
