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
from model import model
from colorize import predict_on_image
import sys


def inference(image_path, model_path):
    
    autoencoder=model()
    
    autoencoder.load_weights(model_path)
    
    prediction = autoencoder.predict(predict_on_image(image_path), verbose=1)
    
    return prediction


image_path=sys.argv[0]
model_save_path=sys.argv[1]

predicted_image=inference(image_path, model_save_path)

cv2.imwrite(predicted_image, './colorized_'+images_path.split('/')[-1])