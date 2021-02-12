# Reference: https://github.com/jacobgil/keras-dcgan

from __future__ import print_function, division

import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Lambda, Dense, LSTM, Activation, Input, Bidirectional, Dropout, Flatten, Dropout, Lambda, BatchNormalization
from tensorflow.keras.layers import Reshape, Conv2DTranspose, TimeDistributed, Conv1D, LeakyReLU, Layer, ReLU, ZeroPadding2D, Conv2D, UpSampling2D
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from os import makedirs

class DCGAN():
    def __init__(self, img_dim, LATENT_DIM, NUM_CHANNELS, TARGET_NAME, TARGET_DF):
        # Input shape
        self.img_rows = img_dim
        self.img_cols = img_dim
        self.latent_dim = LATENT_DIM
        self.channels = NUM_CHANNELS
        self.target_name = TARGET_NAME
        self.target_df = TARGET_DF
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.model_save_dir = 'DCGAN/models/' + self.target_name + '/'

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        
    def save_models(self, epoch):
        self.save_model(self.discriminator,
                        os.path.join(self.model_save_dir, 'discriminator_model' + '_' + str(epoch)))
        self.save_model(self.generator,
                        os.path.join(self.model_save_dir, 'generator_model' + '_' + str(epoch)))
        self.save_model(self.combined,
                        os.path.join(self.model_save_dir, 'combined_model' + '_' + str(epoch)))

    def save_model(self, model, model_path):
        with open(str(model_path) + '.json', 'w') as json_file:
            json_file.write(model.to_json())

        model.save_weights(str(model_path + '.h5'))
        
    # create a line plot of loss for the gan and save to file
    def plot_history(self, d1_hist, d2_hist, g_hist):
        # Convert to arrays, probably a better way
        d1_hist = np.array(d1_hist)
        d2_hist = np.array(d2_hist)
        g_hist = np.array(g_hist)
        # plot history
        plt.figure(figsize=(20,10))
        plt.plot(d1_hist[:,0], label='crit_real')
        plt.plot(d2_hist[:,0], label='crit_fake')
        plt.plot(g_hist, label='gen')
        plt.legend()
        plt.savefig("DCGAN/images/" + self.target_name + '/' + 'plot_line_plot_loss_%s.png' % self.target_name)
        plt.close()

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 9 * 9, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((9, 9, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):
        
        # Create directory to save model results
        if not os.path.isdir(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        # Load the dataset
        X_train = self.target_df.to_numpy().reshape(len(self.target_df), self.img_rows, self.img_cols,1)

        # Rescale -1 to 1
        X_train = 2 * X_train - 1.

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # lists for keeping track of loss
        c1_hist, c2_hist, g_hist = list(), list(), list()
        
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            c1_hist.append(d_loss_real)
            c2_hist.append(d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)
            g_hist.append(g_loss)
            
            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
                self.save_models(epoch)
        
        # line plots of loss
        self.plot_history(c1_hist, c2_hist, g_hist)

    def save_imgs(self, epoch):
        # Create save directory if needed
        img_dir = "DCGAN/images/" + self.target_name + '/'
        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)
        
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        # Store as image
        fig, axs = plt.subplots(r, c, figsize=(10,10))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(img_dir + "mass_spec_im_" + self.target_name.strip() + "%d.png" % epoch)
        plt.close()
        
        # Store as spectrum
        # convert image back to spectrum
        gen_specs_df = pd.DataFrame()
        for img in range(0,len(gen_imgs)):
            gen_specs_df = gen_specs_df.append(pd.Series(gen_imgs[img].reshape(1296)),ignore_index=True)

        fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True, figsize=(15,10))
        x = np.linspace(0,1295,1296)

        for row in ax:
            for col in row:
                col.plot(x, gen_specs_df.loc[random.randrange(24)])

        plt.axis('off')
        fig.savefig(img_dir + "mass_spec_" + self.target_name.strip() + "%d.png" % epoch)
        plt.close()