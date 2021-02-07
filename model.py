"""
building simple GAN models with the following charactristic
gererator : is simple dense models consist of two dense layers
discrimenator :is a simple dense network
by : mahmoud Zaky fetoh, B.Sc.
"""
from distutils.command.build import build

import keras.models as models
import keras.layers as layers
import keras.optimizers as opt
import keras.losses as lss

import numpy as np
import matplotlib.pyplot as plt

output_shape = (28, 28, 1)
z_dim = (100,)


def build_generator(output_shape=output_shape, z_dim=z_dim):
    """
    :param input_shape: is a tuble represent the input dimension i.e.) mnist image is (28,28,1)
    :param z_dim: is the latent space dimension act as generator input
    :return: return a keras model
    two layers of 128 is added with leakyRLU as activation
    function and tanh as activation for output layer
    """
    inp = layers.Input(z_dim, name='generator_input_layer')

    l1 = layers.Dense(128, name='generator_1st_layer')(inp)
    l2 = layers.LeakyReLU(.01, name='generator_1st_acti_layer')(l1)

    l3 = layers.Dense(128, name='generator_2nd_layer')(l2)
    l4 = layers.LeakyReLU(.01, name='generator_2nd_acti_layer')(l3)

    l5 = layers.Dense(np.prod(output_shape), activation='tanh',
                      name='generator_out_layer')(l4)
    l6 = layers.Reshape(output_shape,
                        name='generator_out_reshape_layer')(l5)

    return models.Model(inp, l6)


def build_discriminator(input_shape=output_shape):
    """
    is a simple dense network consist of two layers each 128-size layer
    and the out put layer is a single simoid layer, model perpose to distenguish
    fake example from real example
    :param input_shape: is a tuble if it is image it will be (28,28,1)
    :return: akeras model
    """
    l0 = layers.Input(input_shape, name='discri_input_layer')
    l1 = layers.Flatten(name='discri_Flatten_layer')(l0)

    l2 = layers.Dense(128, name='discri_1st_layer')(l1)
    l3 = layers.LeakyReLU(.01, name='discri_1st_activ_layer')(l2)

    l4 = layers.Dense(128, name='discri_2nd_layer')(l3)
    l5 = layers.LeakyReLU(.01, name='discri_2nd_activ_layer')(l4)

    l6 = layers.Dense(1, activation='sigmoid',
                      name='discri_outlayer')(l5)
    return models.Model(l0, l6)

def compile_GAN(data_dim = output_shape, z_dim = z_dim):
    """
    this function build the GAN model that consist of both discriminator
    and generator
    :param data_dim: the dimension
    :param z_dim: the latent space dimension
    :return: gan, discriminator
    """
    discr = build_discriminator(data_dim)
    discr.compile(opt.adam(),lss.binary_crossentropy,['acc'])

    gener = build_generator(data_dim, z_dim)
    discr.trainable = False
    gan= models.Sequential([gener, discr])
    gan.compile(opt.adam(), lss.binary_crossentropy, ['acc'])
    return gan, discr

def load_model(path= 'gan_model.h5'):
    return models.load_model(path)



if __name__ == '__main__':
    generator = build_generator()
    z_sample = np.random.normal(0, 1, (3,) + z_dim)
    generated = generator.predict(z_sample)
    plt.imshow(np.reshape(generated[0],(28,28)))

    discr = build_discriminator()
    out = discr.predict(generated[0].reshape((1,28,28,1)))
    print(out)
