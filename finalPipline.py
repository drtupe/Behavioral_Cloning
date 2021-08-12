import os
import csv
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout

from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K

## Step 1: Loading Raw Data

lines = []
header = True
camera_images = []
steering_angles = []

with open('driving_log.csv', 'r') as f:
    reader = csv.reader(f, delimiter = ',')
    for row in reader:
        if header:
            header = False
            continue
        steering_center = float(row[3])

        # Steering angle correction factor for stereo cameras
        steering_correction = 0.2
        steering_left = steering_center + steering_correction
        steering_right = steering_center - steering_correction

        # Reading camera images from their paths
        path_src1 = row[0]
        path_src2 = row[1]
        path_src3 = row[2]
        img_name1 = path_src1.split('/')[-1]
        img_name2 = path_src2.split('/')[-1]
        img_name3 = path_src3.split('/')[-1]
        path1 = 'IMG/' + img_name1
        path2 = 'IMG/' + img_name2
        path3 = 'IMG/' + img_name3

        # Image and Steering Dataset
        img_center = np.asarray(Image.open(path1))
        img_left = np.asarray(Image.open(path2))
        img_right = np.asarray(Image.opne(path3))
        camera_images.extend([img_center, img_left, img_right])
        steering_angles.extend([steering_center, steering_left, steering_right])


## Step 2: Dataset Summary & Visual Exploration

print('Number of iamges: ' + str(len(camera_images)))

# Visualizing some random images with their labels

fig, ax = plt.subplots(3, 3, figsize = (16, 8))
fig.subplots_adjust(hspace = 0.5, wspace = 1)
ax = ax.ravel()
for i in range(0, 8, 3):
    # Creating a random idx number that will correspond to a left-cam image
    idx = random.randint(10, len(camera_images))
    idx = idx - (idx % 3) + 1

    # Creating left, center, right images
    img_l = camera_images[idx]
    img_c = camera_images[idx - 1]
    img_r = camera_images[idx + 1]

    # Plotting images
    ax[i].imshow(img_l)
    ax[i].set_title('Left Camera')
    
    ax[i+1].imshow(img_c)
    ax[i+1].set_title('Center Camera')

    ax[i+2].imshow(img_r)
    ax[i+2].set_title('Right Camera')

## Step 3: Data Augmentation

augmentation_imgs, augmented_sas = [], []

for aug_img, aug_sa in zip(camera_images, steering_angles):
    augmented_imgs.append(aug_img)
    augmented_sas.append(aug_sa)

    # Flipping the image
    augmented_imgs.append(cv2.flip(aug_img, 1))

    # Reversing the steering angle for the flipped image
    augmented_sas.append(aug_sa*-1.0)

X_train, y_train = np.array(augmented_imgs), np.array(augmented_sas)
X_train, y_train = np.array(camera_images), np.array(steering_angles)


## Step 4: Pre-Processing the Images
def preprocess(image):
    # Resizing the image
    return tf.image.resize_images(image, (200, 66))


## Step 5: Creating the CNN Architecture

# Keras Sequential Model
model = Sequential()

# Cropping irrelevent parts from image
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))

# Preprocessing the image
model.add(Lambda(preprocess))
model.add(Lambda(lambda x: (x / 127.0 - 1.0)))

# Architecture Layers
model.add(Conv2D(filters = 24, kernel_size = (5, 5), strides = (2, 2), activation = 'relu'))
model.add(Conv2D(filters = 36, kernel_size = (5, 5), strides = (2, 2), activation = 'relu'))
model.add(Conv2D(filters = 48, kernel_size = (5, 5), strides = (2, 2), activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(units = 100, activation = 'relu'))
model.add(Dense(units = 50, activation = 'relu'))
model.add(Dense(units = 10, activation = 'relu'))
model.add(Dense(units = 1))
print(model.summary())


## Step 6: Compiling and Saving the model

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, epoches = 7, validation_split = 0.2, shuffle = True)
model.save('model.h5')
print('Finish..!!')