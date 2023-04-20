# File that performs validation of our PCA-based + encoder-decoder network
# Results are stored in file results_loop.txt
# Source codes by: [Tomasz Hachaj](https://home.agh.edu.pl/~thachaj/)

import matplotlib.pyplot as plt
import keras
from keras import layers
import numpy as np
import math
from shower_generator import *

input_img = keras.Input(shape=(80, 80, 1))
""""""
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.load_weights("./weights/weights.50.h5")
############################################################

import cv2
import random


s = 1.3
N = 1000000#liczba cząstek
r0 = 100#parametr rozkładu

hitX = 800
hitY = 800

file1 = open("results_loop_4.txt", "a")  # append mode
header = "th,phi,th_full,phi_full,th_small,phi_small,th_sample,phi_sample,th_decode,phi_decode"
file1.write(header + "\n")
file1.close()

import time
start = time.time()
for th in range(1, 85, 2):
    for phi in range(1, 180, 2):

        end = time.time()
        print("th=" + str(th) + ", phi=" + str(phi) + ", time=" + str(end - start))
        for a in range(10):
            offsetX = random.randint(0, 12)
            offsetY = random.randint(0, 12)
            #print(offsetX)
            #print(offsetY)

            foo2 = randomvariate(N, xmin=.5, xmax=5 * r0, s=s)
            img = generateArray(foo2, th, phi, hitX, hitY, offsetX, offsetY)

            img_sample = np.zeros(img.shape)

            distance_detectors = 25
            for idx in range(0, img.shape[0], distance_detectors):
                for idy in range(0, img.shape[0], distance_detectors):
                    img_sample[idx, idy] = img[idx, idy]

            blure_size = (int)(4 * distance_detectors + 1)
            img_sample_gausee = cv2.GaussianBlur(img_sample, (blure_size, blure_size), 0)
            img_gausee = cv2.GaussianBlur(img, (blure_size, blure_size), 0)

            img_sample_gausee_small = resizeArray(img_sample_gausee, 10)
            img_gausse_small = resizeArray(img_gausee, 10)

            img_sample_gausee_small_uint8 = convert_to_uint8(img_sample_gausee_small)
            img_gausse_small_uint8 = convert_to_uint8(img_gausse_small)

            img_sample_gausee_small_uint8_scalled = np.copy(img_sample_gausee_small_uint8) / 255.0
            img_sample_gausee_small_uint8_scalled_expand = np.expand_dims(img_sample_gausee_small_uint8_scalled, axis=0)
            #####################################
            decoded_imgs = autoencoder.predict(img_sample_gausee_small_uint8_scalled_expand)
            decoded_imgs = convert_to_uint8(decoded_imgs[0])

            (th_img, phi_img) = computePCA(img, plot_me = False)
            (th_small, phi_small) = computePCA(img_gausse_small_uint8, plot_me = False)
            (th_sample,phi_sample) = computePCA(img_sample_gausee_small_uint8, plot_me = False)
            (th_decoded,phi_decoded) = computePCA(decoded_imgs, plot_me = False)
            line = str(th) + "," + str(phi) + "," + str(th_img) + "," + str(phi_img) + "," \
                    + str(th_small) + "," + str(phi_small) + "," + str(th_sample) + "," + str(phi_sample) + "," + str(th_decoded) + "," + str(phi_decoded)

            file1 = open("results_loop.txt", "a")  # append mode
            file1.write(line + "\n")
            file1.close()
