# File generates data for network training using pre-calculated numpy arrays
# Source codes by: [Tomasz Hachaj](https://home.agh.edu.pl/~thachaj/)

import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt

def resizeArray(img, scale):
    xSize = (int)(img.shape[0] / scale)
    ySize = (int)(img.shape[0] / scale)
    img_res = np.zeros((xSize, ySize))
    for x in range(xSize):
        for y in range(xSize):
            sum = 0
            for xx in range(scale):
                for yy in range(scale):
                    sum = sum + img[(scale * x) + xx, (scale *y) + yy]
            img_res[x,y] = sum
    return img_res

def convert_to_uint8(img):
    img_out = np.copy(img)
    img_out = 255 * ((img_out - np.min(img_out)) / np.max(img_out))
    img_out = img_out.astype(np.uint8)
    return img_out

import glob
import os
# set your path here
all_paths = glob.glob("d:\\dane\\credo_showers\\numpy\\*.npy")
count_max = len(all_paths)
count = 0;

for pp in all_paths:
    print(str(count) + " of " + str(count_max))
    count = count + 1
    file_name = os.path.basename(pp)
    with open(pp, 'rb') as f:
        img = np.load(f)
        xxx = np.zeros(img.shape)

        distance_detectors = 25
        for idx in range(0, img.shape[0], distance_detectors):
            for idy in range(0, img.shape[0], distance_detectors):
                xxx[idx, idy] = img[idx, idy]

        blure_size = (int)(4 * distance_detectors + 1)
        xxx = cv2.GaussianBlur(xxx, (blure_size, blure_size), 0)
        img_gausee = cv2.GaussianBlur(img, (blure_size, blure_size), 0)

        xxx_small = resizeArray(xxx, 10)
        img_gausse_small = resizeArray(img_gausee, 10)

        xxx_small_uint8 = convert_to_uint8(xxx_small)
        img_gausse_small_uint8 = convert_to_uint8(img_gausse_small)

        # set your path here
        cv2.imwrite("d:\\dane\\credo_showers\\sample80x80\\" + file_name + ".png", xxx_small_uint8)
        cv2.imwrite("d:\\dane\\credo_showers\\template80x80\\" + file_name + ".png", img_gausse_small_uint8)
