from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.optimizers import RMSprop
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.models import Sequential
import os
import cv2
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

TRAIN_DIR = 'train_catdogs/train/'
# TEST_DIR = '../input/test/'

ROWS = 64
COLS = 64
CHANNELS = 3

train_dogs = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

train_images = train_dogs[:1000] + train_cats[:1000]
random.shuffle(train_images)
# test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]


def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)  # cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.T
        if i % 250 == 0:
            print('Processed {} of {}'.format(i, count))

    return data


train = prep_data(train_images)
