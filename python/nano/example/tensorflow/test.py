
import os
import pathlib

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.data as tfds
from tensorflow.keras import layers

from bigdl.nano.tf.keras import Sequential

batch_size, img_height, img_width = 32, 256, 256
num_classes = 5

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)

img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20)
gen = img_gen.flow_from_directory(data_dir)

dataset = tf.data.Dataset.from_generator(
  lambda: gen,
  output_types=(tf.float32, tf.float32),
  output_shapes=([None, img_height, img_width, 3], [None, 5])
)


def _fixup_shape(images, labels):
    images.set_shape([None, img_height, img_height, 3])
    labels.set_shape([None, num_classes])
    return images, labels


dataset = dataset.map(_fixup_shape)

AUTOTUNE = tf.data.AUTOTUNE
dataset = dataset.take(100)

model = Sequential([
    layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# work
history = model.fit(dataset, epochs=3)
# not work (for both ray backend and multiprocessing backend)
history = model.fit(dataset, epochs=3, nprocs=2)
