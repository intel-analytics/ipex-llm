import os
import pathlib

import cv2
import numpy as np
import requests
import tensorflow as tf
import tensorflow.python.data as tfds
from tensorflow.keras import layers

from bigdl.nano.tf.keras import Sequential

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32
IMG_HEIGHT, IMG_WIDTH = 256, 256


def dataset_from_generator():
    num_classes = 5

    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)

    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, rotation_range=20)
    gen = img_gen.flow_from_directory(data_dir)

    dataset = tf.data.Dataset.from_generator(
        lambda: gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, IMG_HEIGHT, IMG_HEIGHT, 3], [None, 5])
    ).prefetch(AUTOTUNE)

    return num_classes, dataset


def dataset_from_tensor_slices():
    def preprocess_image(image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [IMG_WIDTH, IMG_WIDTH])
        image /= 255.0  # normalize to [0,1] range

        return image

    def load_and_preprocess_from_path_label(path, label):
        image = tf.io.read_file(path)
        return preprocess_image(image), label


    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

    all_image_paths = list(data_dir.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]

    label_names = sorted(item.name for item in data_dir.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]
    all_image_labels = [tf.one_hot(val, depth=len(label_names)) for val in all_image_labels]

    ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
    image_label_ds = ds.map(load_and_preprocess_from_path_label).shuffle(100).batch(BATCH_SIZE)
    image_label_ds = image_label_ds.prefetch(AUTOTUNE)
    return len(label_names), image_label_ds


def dataset_use_py_function():
    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [IMG_WIDTH, IMG_WIDTH])
        image /= 255.0

        return image

    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

    all_image_paths = list(data_dir.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]

    label_names = sorted(item.name for item in data_dir.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(lambda x: tf.py_function(load_and_preprocess_image,
                                                    inp=[x], Tout=tf.float32),
                           num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    label_ds = label_ds.map(lambda x: tf.one_hot(x, len(label_names)))

    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds)).shuffle(100).batch(BATCH_SIZE)
    image_label_ds = image_label_ds.prefetch(AUTOTUNE)
    return len(label_names), image_label_ds


def dataset_use_np_function():
    def load_and_preprocess_image(path):
        fil = np.array([[-1, -1, 0],
                        [-1, 0, 1],
                        [0, 1, 1]], dtype=np.float32)
        image = cv2.imread(path.decode()).astype(np.float32)
        image = cv2.resize(image, [IMG_WIDTH, IMG_WIDTH])
        image = cv2.filter2D(image, -1, fil)
        image /= 255.0

        return image

    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

    all_image_paths = list(data_dir.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]

    label_names = sorted(item.name for item in data_dir.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]
    all_image_labels = [tf.one_hot(val, depth=len(label_names)) for val in all_image_labels]

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(lambda x: tf.numpy_function(load_and_preprocess_image,
                                                       inp=[x], Tout=tf.float32),
                           num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds)).shuffle(100).batch(BATCH_SIZE)
    image_label_ds = image_label_ds.prefetch(AUTOTUNE)
    return len(label_names), image_label_ds

def model_design(img_height, img_width, num_classes):
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
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    result = {}
    def test_from_generator():
        num_classes, dataset = dataset_from_generator()

        model = model_design(IMG_HEIGHT, IMG_WIDTH, num_classes)
        history = model.fit(dataset, epochs=1, steps_per_epoch=20, batch_size=32)

        try:
            model = model_design(IMG_HEIGHT, IMG_WIDTH, num_classes)
            history = model.fit(dataset, epochs=1, steps_per_epoch=20, nprocs=2)
        except:
            result["generator"] = False
        else:
            result["generator"] = True


    def test_from_tensor_slices():
        num_classes, dataset = dataset_from_tensor_slices()

        model = model_design(IMG_HEIGHT, IMG_WIDTH, num_classes)
        history = model.fit(dataset, epochs=1, steps_per_epoch=20, batch_size=32)

        try:
            model = model_design(IMG_HEIGHT, IMG_WIDTH, num_classes)
            history = model.fit(dataset, epochs=1, steps_per_epoch=20, nprocs=2)
        except:
            result["tensor_slice"] = False
        else:
            result["tensor_slice"] = True

    def test_map_py_function():
        num_classes, dataset = dataset_use_py_function()

        model = model_design(IMG_HEIGHT, IMG_WIDTH, num_classes)
        history = model.fit(dataset, epochs=1, steps_per_epoch=20, batch_size=32)

        try:
            model = model_design(IMG_HEIGHT, IMG_WIDTH, num_classes)
            history = model.fit(dataset, epochs=1, steps_per_epoch=20, nprocs=2)
        except:
            result["py_function"] = False
        else:
            result["py_function"] = True

    def test_map_np_function():
        num_classes, dataset = dataset_use_np_function()

        model = model_design(IMG_HEIGHT, IMG_WIDTH, num_classes)
        history = model.fit(dataset, epochs=1, steps_per_epoch=20)

        try:
            model = model_design(IMG_HEIGHT, IMG_WIDTH, num_classes)
            history = model.fit(dataset, epochs=1, steps_per_epoch=20, nprocs=2)
        except:
            result["numpy_function"] = False
        else:
            result["numpy_function"] = True

    test_from_generator()
    test_from_tensor_slices()
    test_map_py_function()
    test_map_np_function()
    print("===================Test Results=======================")
    for key, val in result.items():
        print(f"{key}: {val}")
