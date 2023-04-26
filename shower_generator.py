# File containing implementations of auxiliary functions
# Source codes by: 

import math
import numpy as np
from numba import njit, prange
import scipy.special as sp
import cv2
import matplotlib.pyplot as plt
import random as rn
###############################################################################################
from math import atan2, cos, sin, sqrt, pi

def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

def getOrientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))

    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (
    cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (
    cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)
    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians

    return angle

def calculatPol(x, y, exp):
    th = math.acos(exp[1] / exp[0])
    phi = math.pi - np.arctan2(y, x)
    return np.array((th, phi))

def computePCA(img, plot_me = False):
    img = np.copy(img)

    img = 255 * ((img - np.min(img)) / np.max(img))
    img = img.astype(np.uint8)

    rng = np.random.RandomState(1)
    points_img = list()
    for a in range(img.shape[0]):
        for b in range(img.shape[1]):
            if img[a,b] > 5:
                for c in range(int(img[a,b])):
                    #points_img.append((a, b))
                    points_img.append((a, b))
    XX = np.asarray(points_img)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(XX)

    if plot_me:
        for a in range(-10, 10, 1):
            for b in range(-10, 10, 1):
                img[int(pca.mean_[0]) + a,
                    int(pca.mean_[1]) + b] = 255

                v = pca.components_[0] * 3 * np.sqrt(pca.explained_variance_[0])

                img[int(pca.mean_[0]) + a + int(v[0]),
                    int(pca.mean_[1]) + b + int(v[1])] = 255

                v = pca.components_[1] * 3 * np.sqrt(pca.explained_variance_[1])

                img[int(pca.mean_[0]) + a + int(v[0]),
                    int(pca.mean_[1]) + b + int(v[1])] = 255

        plt.imshow(img, cmap='hot')
        plt.axis('equal');
        plt.show()

    exp = np.sqrt(pca.explained_variance_) / np.sum(np.sqrt(pca.explained_variance_))
    v = pca.components_[0] * exp[1] / exp[0]
    if v[0] < 0:
        v = -1 * v

    res_pca = calculatPol(v[1], v[0], exp) * 180 / math.pi
    return (res_pca[0], res_pca[1])



###################################################################################

def ro(r, s, N, r0):
    return N/(2*np.pi*r0**2)*sp.gamma(4.5-s)/(sp.gamma(s)*sp.gamma(4.5-2*s))*(r/r0)**(s-2)*(1+r/r0)**(s-4.5)

def randomvariate(n=1000, xmin=0, xmax=1, s=1.3, r0=100):
    # Calculates the minimal and maximum values of the PDF in the desired
    # interval. The rejection method needs these values in order to work
    # properly.

    xminHelp = xmin
    xmin = 0
    x = np.linspace(xmin, xmax, 1000)
    y = ro(x, s, n, r0)
    y[x <= xminHelp] = ro(xminHelp, s, n, r0)
    pmin = 0.
    pmax = y.max()

    ran = np.zeros((n))
    ran = generate_sample(ran, xmin, xmax, pmin, pmax, s, n, r0)
    return ran

@njit(parallel=True)
def generate_sample(ran, xmin, xmax, pmin, pmax, s, N, r0):
    for aa in prange(ran.shape[0]):
        accept = False
        x = 0
        while not(accept):
            x = np.random.uniform(xmin, xmax)  # x'
            y = np.random.uniform(pmin, pmax)  # y'

            r = x
            v = N / (2 * np.pi * r0 ** 2) * math.gamma(4.5 - s) / (math.gamma(s) * math.gamma(4.5 - 2 * s)) * (r / r0) ** (
                        s - 2) * (1 + r / r0) ** (s - 4.5)
            if y < v:
                accept = True
        ran[aa] = x
    return ran

def generateArray(rndr, th, phi, hitX, hitY, offsetX = 0, offsetY = 0):

    (x, y) = generateRotation(rndr)
    th_ = th * np.pi / 180
    phi_ = phi * np.pi / 180

    x = x / np.cos(th_)
    
    cosPhi = np.cos(phi_)
    sinPhi = np.sin(phi_)
    xr = x * cosPhi - y * sinPhi
    yr = x * sinPhi + y * cosPhi

    #wynik będzie w centrymetrach, wyniki z rndr są w metrach
    backgroundMezonsPerSquaredCentimeter = 1.0 / 60
    backgroundMionsCount = int(hitX * hitY * backgroundMezonsPerSquaredCentimeter)

    img = np.zeros([int(hitX), int(hitY)])
    # tło
    for a in range(backgroundMionsCount):
        x = int(np.random.randint(0, img.shape[0], 1))
        y = int(np.random.randint(0, img.shape[1], 1))
        img[x, y] = img[x, y] + 1
    # mapowanie rotacji
    hitX2 = hitX / 2.0
    hitY2 = hitY / 2.0
    for a in range(xr.shape[0]):
        xx = int(hitX2 - (100.0 * xr[a] + offsetX))
        yy = int(hitY2 + (100.0 * yr[a] + offsetY))
        if xx >= 0 and xx < hitX and yy > 0 and yy < hitY:
            img[yy, xx] = img[yy, xx] + 1
    return img

@njit(parallel=True)
def generateRotation(rndr):
    xx = np.zeros(rndr.shape[0])
    yy = np.zeros(rndr.shape[0])
    for a in prange(rndr.shape[0]):
        r = rndr[a]
        phi_in = rn.uniform(0, 2 * np.pi)
        x = r * np.cos(phi_in)
        y = r * np.sin(phi_in)
        xx[a] = x
        yy[a] = y
    return (xx, yy)

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

