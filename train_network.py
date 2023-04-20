# Training a deep encoder-decoder using generated data
# File containing implementations of auxiliary functions

import keras
from keras import layers

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

import tensorflow as tf
ada = tf.keras.optimizers.Adam(
    learning_rate=0.001)
autoencoder.compile(optimizer=ada, loss='binary_crossentropy')
autoencoder.summary()

checkpoint_filepath = "./weights/weights.{epoch:02d}.h5"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    #mode='max',
    save_best_only=False)
from keras.callbacks import CSVLogger
csv_logger = CSVLogger('./log_tran.csv', append=True, separator=';')

from keras.preprocessing.image import ImageDataGenerator
seed = 0

image_datagen = ImageDataGenerator(
        rescale=1./255)
image_datagen2 = ImageDataGenerator(
        rescale=1./255)


batch_size = 128
image_generator = image_datagen.flow_from_directory(
    'd:\\dane\\credo_showers\\sample80x80_tran\\',
    color_mode='grayscale',
    target_size=(80, 80),
    class_mode=None,
    batch_size=batch_size,
    seed=seed)
mask_generator = image_datagen2.flow_from_directory(
    'd:\\dane\\credo_showers\\template80x80_tran\\',
    color_mode='grayscale',
    target_size=(80, 80),
    class_mode=None,
    batch_size=batch_size,
    seed=seed)
# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

autoencoder.fit(
    train_generator,
    epochs=50,
    steps_per_epoch = image_generator.n//image_generator.batch_size,
    #batch_size=128,
    shuffle=True,
    callbacks=[model_checkpoint_callback, csv_logger])
