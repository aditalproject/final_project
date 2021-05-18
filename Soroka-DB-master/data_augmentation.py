# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 13:08:16 2019

@author: Mor
"""

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# declare db path
mammos_path = os.path.join(os.getcwd(), 'DB', 'mammos.npy')
labels_path = os.path.join(os.getcwd(), 'DB', 'labels.npy')

# read the existing data and labels
x_train = np.load(mammos_path)
y_train = np.load(labels_path)

# create the data generator
datagen = ImageDataGenerator(
    rotation_range=90, #A rotation augmentation randomly rotates the image clockwise by a given number of degrees from 0 to 90.
    width_shift_range=0.2, #The percentage (between 0 and 1) of the width of the image to shift
    height_shift_range=0.2, #The percentage (between 0 and 1) of the height of the image to shift
    horizontal_flip=True, #An image flip means reversing the rows or columns of pixels in the case of a vertical or horizontal flip respectively.
    vertical_flip=True)

# fit it to the data
datagen.fit(x_train, augment=True)

# concatenate 
counter = 1

datagen_iterator = datagen.flow(x_train, y_train, batch_size=x_train.shape[0])

for x_batch, y_batch in datagen_iterator:
    x_train = np.concatenate((x_train, x_batch), axis=0)
    y_train = np.concatenate((y_train, y_batch), axis=0)
    counter += 1
    if counter == 5: # end after 5 variations
        break

np.save('x_train', np.squeeze(x_train))
np.save('y_train', np.squeeze(y_train))
