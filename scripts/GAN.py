from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from PIL import Image
import numpy
import glob
import matplotlib.image as mpimg

import sys
import numpy as np

class GAN():
    def __init__(self):
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 256*256*3

        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.generator = self.build_generator()

        z = Input(shape=(self.latent_dim,)) #input data
        img = self.generator(z) #generate image from Z

        self.discriminator.trainable = False #?

        validity = self.discriminator(img) #this is the discriminator set on the generated image from Z

        self.combined = Model(z, validity) #merge both models thus the input, and validity model!
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='sigmoid')) #tanh because it is between 0 and 1
        model.add(Reshape(self.img_shape))
        model.summary()
        noise = Input(shape=(self.latent_dim,))
        img = model(noise) #create an image from the noise
        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img) #sigmoid if it is real or not

        return Model(img, validity)

    def import_sample_IDs(self, directory = '/Users/Hidde/kaggle/monet_jpg/'):
        fileNames = numpy.array(glob.glob(directory + '*.jpg'))
        return fileNames

    def import_training_data(self, batch_size, fileNames, type):
        trainingFiles=list(fileNames[i] for i in np.random.randint(0, len(fileNames), batch_size))

        trainingImages = []
        for i in trainingFiles:
            if type == 'painting':
                im = np.array(Image.open(i))
                trainingImages.append(im)

            if type == 'photo':
                im = np.array(Image.open(i))
                im = im.flatten(order = 'C')
                trainingImages.append(im)
        return np.array(trainingImages)

    def train(self, epochs, batch_size=128, sample_interval=50):
        fileNamesPainting = self.import_sample_IDs(directory = '/Users/Hidde/kaggle/monet_jpg/')
        fileNamesPhotos = self.import_sample_IDs(directory = '/Users/Hidde/kaggle/photo_jpg/')

        #(X_train, _), (_, _) = mnist.load_data() #this have to be the monet pictures
        #X_train = X_train / 127.5 - 1.
        #X_train = np.expand_dims(X_train, axis=3)
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            #idx = np.random.randint(0, X_train.shape[0], batch_size) #get random sample
            #imgs = X_train[idx]
            #print(imgs.shape)
            imgs = self.import_training_data(batch_size = batch_size, fileNames = fileNamesPainting, type = 'painting')
            #noise = np.random.normal(0, 1, (batch_size, self.latent_dim)) #random noise for the generator, real photos
            noise = self.import_training_data(batch_size = batch_size, fileNames = fileNamesPhotos, type = 'photo')
            gen_imgs = self.generator.predict(noise) #genrate images from the noise using the untrained generator.
            d_loss_real = self.discriminator.train_on_batch(imgs, valid) #train the discriminator on the real images, al 1s
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake) #train  the discriminator on the false zeroes, al 0s
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) #take the average of the two loss function on the two sets

            #noise = np.random.normal(0, 1, (batch_size, self.latent_dim)) #again subsample of real photos.
            noise = self.import_training_data(batch_size = batch_size, fileNames = fileNamesPhotos, type = 'photo')

            g_loss = self.combined.train_on_batch(noise, valid) #train the generator with novel noise, which is used to fool the discriminator (has to say if the noise (Z) is real)
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
            if epoch % sample_interval == 0:
                self.sample_images(epoch, fileNames = fileNamesPhotos)
                self.combined.save("/Users/Hidde/kaggle/models/combined_%d.md" % epoch)
                self.generator.save("/Users/Hidde/kaggle/models/generator_%d.md" % epoch)
                self.discriminator.save("/Users/Hidde/kaggle/models/discriminator_%d.md" % epoch)

    def sample_images(self, epoch, fileNames):
        #r, c = 256, 256
        #noise = np.random.normal(0, 1, (r * c, self.latent_dim)) #real photos
        noise = self.import_training_data(batch_size=1, fileNames=fileNames, type = 'photo')
        gen_imgs = self.generator.predict(noise)
        gen_imgs = gen_imgs[0]
        gen_imgs = gen_imgs*255
        gen_imgs = gen_imgs.astype(np.uint8)
        im = Image.fromarray(gen_imgs)
        im.save("/Users/Hidde/kaggle/images/%d.png" % epoch)



if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=500, batch_size=128, sample_interval=10)
