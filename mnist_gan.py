# This code will be used to build a Generative Adversarial Network 
# for generating handwritten digits

from tensorflow.keras.datasets import mnist

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import initializers

import matplotlib.pyplot as plt
import numpy as np

randomDim = 10
(X_train, _), (_, _) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5

# Using a multi-layered perceptron an image wiht a size of 784
# and 60,000 training images
X_train = X_train.reshape(60000, 784)

