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

# Building a generator
generator = Sequential() 

# Adding 256 neurons
generator.add(Dense(256, input_dim=randomDim))

# Adding a Leaky Rectified Linear Unit
generator.add(LeakyReLU(0.2))

# Adding a 512 neuron hidden layer
generator.add(Dense(512))

# And another Leaky ReLU
generator.add(LeakyReLU(0.2))

# Adding another hidden layer with 1024 neurons
generator.add(Dense(1024))

# Yet another Leaky ReLU
generator.add(LeakyReLU(0.2))

# Finally, an output layer of 784 neurons
generator.add(Dense(784, activation='tanh'))


# Now, we build the discrimiator
discriminator = Sequential()

# The input size for this discriminator will be 784
discriminator.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)))

# Adding a Leaky ReLU
discriminator.add(LeakyReLU(0.2))

# Add a dropout
discriminator.add(Dropout(0.3))

discriminator.add(Dense(512))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))

discriminator.add(Dense(256))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))

discriminator.add(Dense(1, activation='sigmoid'))

# Combine the generator and the discriminator to form a GAN
discriminator.trainable = False
ganInput = Input (shape=(randomDim,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)

# Freeze the weights of the discriminator
discriminator.compile(loss='binary_crossentropy', optimizer='adam')
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Begin training the discriminator
def train(epochs=1, batchSize=128):
    batchCount = int(X_train.shape[0] / batchSize)
    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)
    for e in range(1, epochs+1):
        print('-'*15, 'Epoch %d' % e, '-'*15)
        # Get a random set of random noise and images
        for _ in range(batchCount):
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            # Generate fake MNIST images
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize)]
            generatedImages = generator.predict(noise)
            # Labels for generated and real data
            X = np.concatenate([imageBatch, generatedImages])
            # One-sided label smoothing
            yDis = np.zeros(2*batchSize)
            # Train the discriminator
            yDis[:batchSize]
            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)
            # Now to train the generator
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            yGen = np.ones(batchSize)
            discriminator.trainable = False
            gloss = gan.train_on_batch(noise, yGen)
            # Store the loss of hte most recent batch from this epoch
            dLosses.append(dloss)
            gLosses.append(gloss)
            if e == 1 or e % 20 == 0:
                saveGeneratedImages(e)



# Plot hte loss from each batch
def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminative loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/gan_loss_epoch_%d.png' % epoch)
    
# Create an Array of generated MNIST images
def saveGeneratedImages(epoch, examples=100, dim=(10,10), figsize=(10,10)):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    