# The file visualizes an example of how of our PCA-based + encoder-decoder metod works for selected parameters
# Source codes by: [Tomasz Hachaj](https://home.agh.edu.pl/~thachaj/)

from shower_generator import *

import keras
from keras import layers

def computePCA2(img):
    img = np.copy(img)

    img = 255 * ((img - np.min(img)) / np.max(img))
    img = img.astype(np.uint8)

    points_img = list()
    for a in range(img.shape[0]):
        for b in range(img.shape[1]):
            if img[a,b] > 5:
                for c in range(int(img[a,b])):
                    points_img.append((a, b))
    XX = np.asarray(points_img)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(XX)

    exp = np.sqrt(pca.explained_variance_) / np.sum(np.sqrt(pca.explained_variance_))
    v = pca.components_[0] * exp[1] / exp[0]
    if v[0] < 0:
        v = -1 * v

    res_pca = calculatPol(v[1], v[0], exp) * 180 / math.pi

    v0 = pca.components_[0] * 2 * np.sqrt(pca.explained_variance_[0])
    v1 = pca.components_[1] * 2 * np.sqrt(pca.explained_variance_[1])

    v0 = v0 if v0[0] > 0 else -1 * v0
    v1 = v1 if v1[0] > 0 else -1 * v1

    return (res_pca[0], res_pca[1], pca.mean_, v0, v1)

input_img = keras.Input(shape=(80, 80, 1))

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

# update path here
autoencoder.load_weights("./weights/weights.50.h5")

plot_pca = True
th = 73
phi = 145

s = 1.3
N = 1000000#liczba cząstek
r0 = 100#parametr rozkładu

offsetX = 5
offsetY = 7

hitX = 800
hitY = 800

import time

start = time.time()
foo2 = randomvariate(N, xmin=.5, xmax=5 * r0, s=s, r0 = r0)
img = generateArray(foo2, th, phi, hitX, hitY, offsetX, offsetY)
end = time.time()
print("time=" + str(end - start))


img_res = img
xxx = np.zeros(img_res.shape)

distance_detectors = 25
for idx in range(0, img_res.shape[0], distance_detectors):
    for idy in range(0, img_res.shape[0], distance_detectors):
        xxx[idx, idy] = img_res[idx, idy]

import cv2
blure_size = (int)(4 * distance_detectors + 1)
xxx = cv2.GaussianBlur(xxx,(blure_size,blure_size),0)
img_gausee = cv2.GaussianBlur(img,(blure_size,blure_size),0)


#computePCA(img_gausee)

def convert_to_uint8(img):
    img_out = np.copy(img)
    img_out = 255 * ((img_out - np.min(img_out)) / np.max(img_out))
    img_out = img_out.astype(np.uint8)
    return img_out


print("th=" + str(th) + ", ph=" + str(phi))

fig=plt.figure()
p1 = fig.add_subplot(2,3,1)

if plot_pca:
    (th,phi, pca_mean, v0, v1) = computePCA2(img)
    print("th=" + str(th) + ", ph=" + str(phi))
    p1.arrow(pca_mean[1],pca_mean[0],v0[1], v0[0], fc="g", ec="g", head_width=30, head_length=30)
    p1.arrow(pca_mean[1],pca_mean[0],v1[1], v1[0], fc="cyan", ec="cyan", head_width=30, head_length=30)

imgplot = plt.imshow(img, cmap='hot')
p1.title.set_text('Img')
plt.colorbar()

p2 = fig.add_subplot(2,3,2)

if plot_pca:
    (th,phi, pca_mean, v0, v1) = computePCA2(img_gausee)
    print("th=" + str(th) + ", ph=" + str(phi))
    p2.arrow(pca_mean[1],pca_mean[0],v0[1], v0[0], fc="g", ec="g", head_width=30, head_length=30)
    p2.arrow(pca_mean[1],pca_mean[0],v1[1], v1[0], fc="cyan", ec="cyan", head_width=30, head_length=30)

imgplot = plt.imshow(img_gausee, cmap='hot')
p2.title.set_text('Img + Gauss')
plt.colorbar()

img_res_resize = resizeArray(img_gausee, 10)
p3 = fig.add_subplot(2,3,3)

if plot_pca:
    (th,phi, pca_mean, v0, v1) = computePCA2(img_res_resize)
    print("th=" + str(th) + ", ph=" + str(phi))
    p3.arrow(pca_mean[1],pca_mean[0],v0[1], v0[0], fc="g", ec="g", head_width=3, head_length=3)
    p3.arrow(pca_mean[1],pca_mean[0],v1[1], v1[0], fc="cyan", ec="cyan", head_width=3, head_length=3)

imgplot = plt.imshow(img_res_resize, cmap='hot')
p3.title.set_text('Img + Gauss + resampled')
plt.colorbar()

p4 = fig.add_subplot(2,3,4)

if plot_pca:
    (th,phi, pca_mean, v0, v1) = computePCA2(xxx)
    print("th=" + str(th) + ", ph=" + str(phi))
    p4.arrow(pca_mean[1],pca_mean[0],v0[1], v0[0], fc="g", ec="g", head_width=30, head_length=30)
    p4.arrow(pca_mean[1],pca_mean[0],v1[1], v1[0], fc="cyan", ec="cyan", head_width=30, head_length=30)

imgplot = plt.imshow(xxx, cmap='hot')
p4.title.set_text('Img + sampled + Gauss')
plt.colorbar()

img_res_xxx = resizeArray(xxx, 10)

p5 = fig.add_subplot(2,3,5)

if plot_pca:
    (th,phi, pca_mean, v0, v1) = computePCA2(img_res_xxx)
    print("th=" + str(th) + ", ph=" + str(phi))
    p5.arrow(pca_mean[1],pca_mean[0],v0[1], v0[0], fc="g", ec="g", head_width=3, head_length=3)
    p5.arrow(pca_mean[1],pca_mean[0],v1[1], v1[0], fc="cyan", ec="cyan", head_width=3, head_length=3)

imgplot = plt.imshow(img_res_xxx, cmap='hot')
p5.title.set_text('Img + sampled + Gauss + resampled')
plt.colorbar()

#############################################################
img_res_xxx_uint8 = convert_to_uint8(img_res_xxx)
img_sample_gausee_small_uint8_scalled = np.copy(img_res_xxx_uint8) / 255.0
img_sample_gausee_small_uint8_scalled_expand = np.expand_dims(img_sample_gausee_small_uint8_scalled, axis=0)
decoded_imgs = autoencoder.predict(img_sample_gausee_small_uint8_scalled_expand)
decoded_imgs = convert_to_uint8(decoded_imgs[0])
#############################################################

p6 = fig.add_subplot(2,3,6)

if plot_pca:
    (th,phi, pca_mean, v0, v1) = computePCA2(decoded_imgs)
    print("th=" + str(th) + ", ph=" + str(phi))
    p6.arrow(pca_mean[1],pca_mean[0],v0[1], v0[0], fc="g", ec="g", head_width=3, head_length=3)
    p6.arrow(pca_mean[1],pca_mean[0],v1[1], v1[0], fc="cyan", ec="cyan", head_width=3, head_length=3)

imgplot = plt.imshow(decoded_imgs, cmap='hot')
p6.title.set_text('Img + sampled + Gauss + resampled + E-D')
plt.colorbar()

plt.show()
