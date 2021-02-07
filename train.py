import numpy as np
import utility as util

z_dim= 100
img_shape= (28,28,1)

def train(gan_models, epoch, batch_size, show_sample= 100 ):
    gan, discr = gan_models
    gener = gan.layers[0]
    img_train, img_test = util.get_prep_mnist()
    fake = np.zeros((batch_size,1), dtype= np.float)
    real = np.ones((batch_size,1), dtype= np.float)

    for _ in range(epoch):
        ##Training the discriminator
        indx_sample = np.random.randint(0,img_train.shape[0], batch_size)
        z_samples = np.random.normal(0,1,(batch_size,z_dim))
        real_imgs = img_train[indx_sample]
        fake_imgs = gener.predict(z_samples)
        real_loss = discr.train_on_batch(real_imgs, real)
        fake_loss = discr.train_on_batch(fake_imgs, fake)
        #discr_avr_loss = (real_loss + fake_loss)/2
        ##Training the Generator
        z_samples = np.random.normal(0, 1, (batch_size, z_dim))
        gan_loss = gan.train_on_batch(z_samples, real)
        print(_)
        #if (_+1) % show_sample == 0 :
            #util.plot(gener)


