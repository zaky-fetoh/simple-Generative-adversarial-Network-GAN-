"""
this module provide the require function
by : mahmoud Zaky fetoh, B.Sc.
"""

import keras.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

def noramlize_and_reshape(X):
    X = X / 127.5 - 1
    X = np.reshape(X, X.shape + (1,))
    return X

def get_prep_mnist():
    (X_train,Y_train),(X_test,Y_test) = datasets.mnist.load_data()
    return noramlize_and_reshape(X_train) , noramlize_and_reshape(X_test)


def plot(gener, row=4, col=4):
    num = row * col
    z_sample = np.random.normal(0,1,(num,100))
    imgs = gener.predict(z_sample)
    imgs.shape = imgs.shape[:-1]
    for i in range(num):
        plt.subplot(row,col,i+1)
        plt.imshow(imgs[i], cmap= 'gray')
        plt.axis('off')
