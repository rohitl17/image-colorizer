import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.layers import Dense, Input, Conv2D, Conv2DTranspose, Flatten, Reshape
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
import cv2
from sklearn.metrics import accuracy_score
import glob, random


def model():
    input_shape = (256, 256, 1)
    batch_size = 32
    kernel_size = 3
    latent_dim = 256
    channels=3

    # encoder/decoder number of CNN layers and filters per layer
    layer_filters = [64, 128, 256]

    # Autoencoder model
    # Building the encoder
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    # Stack of Conv2D(64)-Conv2D(128)-Conv2D(256)
    for filters in layer_filters:
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=2,
                   activation='relu',
                   padding='same')(x)

    shape = K.int_shape(x)

    # Generate a latent vector
    x = Flatten()(x)
    latent = Dense(latent_dim, name='latent_vector')(x)

    # instantiate encoder model
    encoder = Model(inputs, latent, name='encoder')
    # encoder.summary()

    # build the decoder model
    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    # stack of Conv2DTranspose(256)-Conv2DTranspose(128)-Conv2DTranspose(64)
    for filters in layer_filters[::-1]:
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=2,
                            activation='relu',
                            padding='same')(x)

    outputs = Conv2DTranspose(filters=channels,
                              kernel_size=kernel_size,
                              activation='sigmoid',
                              padding='same',
                              name='decoder_output')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    #decoder.summary()

    # autoencoder = encoder + decoder
    # instantiate autoencoder model
    autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    
    return autoencoder