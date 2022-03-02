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
NUM_CLASSES = 5


def dataset_from_generator():
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)

    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, rotation_range=20)
    gen = img_gen.flow_from_directory(data_dir)

    dataset = tf.data.Dataset.from_generator(
        lambda: gen,
        output_signature=(tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_HEIGHT, 3),
                                              dtype=tf.float32),
                          tf.TensorSpec(shape=(None, 5), dtype=tf.float32))
    ).prefetch(AUTOTUNE)

    return dataset


def dataset_from_tensor_slices():
    def preprocess_image(image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [IMG_WIDTH, IMG_WIDTH])
        image /= 255.0  # normalize to [0,1] range

        return image

    def load_and_preprocess_from_path_label(path, label):
        image = tf.io.read_file(path)
        return preprocess_image(image), label

    def dataset_fn(ds):
        return ds.map(load_and_preprocess_from_path_label)

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
    image_label_ds = ds.apply(dataset_fn).shuffle(100).batch(BATCH_SIZE)
    image_label_ds = image_label_ds.prefetch(AUTOTUNE)
    return image_label_ds


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
    return image_label_ds


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

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(lambda x: tf.numpy_function(load_and_preprocess_image,
                                                       inp=[x], Tout=tf.float32),
                           num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    label_ds = label_ds.map(lambda x: tf.one_hot(x, len(label_names)))

    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds)).shuffle(100).batch(BATCH_SIZE)
    image_label_ds = image_label_ds.prefetch(AUTOTUNE)
    return image_label_ds


def generate_samples_labels():
    train_examples = np.random.random((100, IMG_HEIGHT, IMG_HEIGHT, 3))
    train_labels = np.eye(NUM_CLASSES)[np.random.randint(0, NUM_CLASSES, size=(100,))]
    return train_examples, train_labels


def dataset_choice_dataset():
    train_examples, train_labels = generate_samples_labels()
    train_datasets = [tf.data.Dataset.from_tensor_slices((train_examples, train_labels)).repeat(),
                      tf.data.Dataset.from_tensor_slices((train_examples, train_labels)).repeat(),
                      tf.data.Dataset.from_tensor_slices((train_examples, train_labels)).repeat()]
    choice_dataset = tf.data.Dataset.range(3).repeat(2)
    return tf.data.Dataset.choose_from_datasets(train_datasets, choice_dataset).batch(BATCH_SIZE)


def dataset_concatenate():
    train_examples, train_labels = generate_samples_labels()
    train_dataset1 = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    train_dataset2 = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    return train_dataset1.concatenate(train_dataset2).batch(BATCH_SIZE)


def dataset_filter():
    train_examples, train_labels = generate_samples_labels()
    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    return train_dataset.filter(lambda x, y: tf.reduce_sum(x) > 50).repeat().batch(BATCH_SIZE)


def dataset_apply():
    def dataset_fn(ds):
        return ds.filter(lambda x, y: tf.reduce_sum(x) > 50)
    train_examples, train_labels = generate_samples_labels()
    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    return train_dataset.apply(dataset_fn).repeat().repeat().batch(BATCH_SIZE)


def dataset_flat_map():
    train_examples1, train_labels1 = generate_samples_labels()
    train_examples2, train_labels2 = generate_samples_labels()
    train_examples = tf.data.Dataset.from_tensor_slices([train_examples1, train_examples2])
    train_labels = tf.data.Dataset.from_tensor_slices([train_labels1, train_labels2])
    train_ds = train_examples.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x)).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    train_label = train_labels.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x)).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return train_ds.as_numpy_iterator(), train_label.as_numpy_iterator()

def model_design(img_height, img_width):
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
        layers.Dense(NUM_CLASSES)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    result = {}
    def test_from_generator():
        dataset = dataset_from_generator()
        model = model_design(IMG_HEIGHT, IMG_WIDTH)

        try:
            history = model.fit(dataset, epochs=1, steps_per_epoch=20, nprocs=2)
        except:
            result["generator"] = False
        else:
            result["generator"] = True


    def test_from_tensor_slices():
        dataset = dataset_from_tensor_slices()
        model = model_design(IMG_HEIGHT, IMG_WIDTH)

        try:
            history = model.fit(dataset, epochs=1, steps_per_epoch=20, nprocs=2)
        except:
            result["tensor_slice"] = False
        else:
            result["tensor_slice"] = True

    def test_map_py_function():
        dataset = dataset_use_py_function()
        model = model_design(IMG_HEIGHT, IMG_WIDTH)

        try:
            history = model.fit(dataset, epochs=1, steps_per_epoch=20, nprocs=2)
        except:
            result["py_function"] = False
        else:
            result["py_function"] = True

    def test_map_np_function():
        dataset = dataset_use_np_function()
        model = model_design(IMG_HEIGHT, IMG_WIDTH)

        try:
            history = model.fit(dataset, epochs=1, steps_per_epoch=20, nprocs=2)
        except:
            result["numpy_function"] = False
        else:
            result["numpy_function"] = True

    def test_choice_dataset():
        dataset = dataset_choice_dataset()
        model = model_design(IMG_HEIGHT, IMG_WIDTH)

        try:
            history = model.fit(dataset, epochs=1, steps_per_epoch=20, nprocs=2)
        except:
            result["choice_dataset"] = False
        else:
            result["choice_dataset"] = True

    def test_concatenate():
        dataset = dataset_concatenate()
        model = model_design(IMG_HEIGHT, IMG_WIDTH)

        try:
            history = model.fit(dataset, epochs=1, steps_per_epoch=20, nprocs=2)
        except:
            result["concatenate"] = False
        else:
            result["concatenate"] = True

    def test_apply():
        dataset = dataset_apply()
        model = model_design(IMG_HEIGHT, IMG_WIDTH)

        try:
            history = model.fit(dataset, epochs=1, steps_per_epoch=20, nprocs=2)
        except:
            result["apply"] = False
        else:
            result["apply"] = True

    def test_filter():
        dataset = dataset_filter()
        model = model_design(IMG_HEIGHT, IMG_WIDTH)

        try:
            history = model.fit(dataset, epochs=1, steps_per_epoch=20, nprocs=2)
        except:
            result["filter"] = False
        else:
            result["filter"] = True

    def test_flat_map():
        dataset = dataset_flat_map()
        model = model_design(IMG_HEIGHT, IMG_WIDTH)

        try:
            history = model.fit(dataset, epochs=1, steps_per_epoch=20, nprocs=2)
        except:
            result["flat_map"] = False
        else:
            result["flat_map"] = True

    test_from_generator()
    test_from_tensor_slices()
    test_map_py_function()
    test_map_np_function()
    test_filter()
    test_concatenate()
    test_apply()
    test_choice_dataset()
    # test_flat_map()
    print("===================Test Results=======================")
    for key, val in result.items():
        print(f"{key}: {val}")
