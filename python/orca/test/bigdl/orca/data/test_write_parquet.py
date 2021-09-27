#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import tempfile
import shutil

import numpy as np
import os
from bigdl.orca.data.image.parquet_dataset import ParquetDataset
from bigdl.orca.data.image.parquet_dataset import _write_ndarrays, write_from_directory, write_parquet
from bigdl.orca.data.image.utils import DType, FeatureType, SchemaField
from bigdl.orca.learn.tf.estimator import Estimator
from bigdl.orca.data.image import write_mnist, write_voc

resource_path = os.path.join(os.path.split(__file__)[0], "../resources")


def test_write_parquet_simple(orca_context_fixture):
    sc = orca_context_fixture
    temp_dir = tempfile.mkdtemp()

    def generator(num):
        for i in range(num):
            yield {"id": i, "feature": np.zeros((10,)), "label": np.ones((4,))}

    schema = {
        "id": SchemaField(feature_type=FeatureType.SCALAR, dtype=DType.INT32, shape=()),
        "feature": SchemaField(feature_type=FeatureType.NDARRAY, dtype=DType.FLOAT32, shape=(10,)),
        "label": SchemaField(feature_type=FeatureType.NDARRAY, dtype=DType.FLOAT32, shape=(4,))
    }

    try:

        ParquetDataset.write("file://" + temp_dir, generator(100), schema)
        data, schema = ParquetDataset._read_as_dict_rdd("file://" + temp_dir)
        data = data.collect()[0]
        assert data['id'] == 0
        assert np.all(data['feature'] == np.zeros((10,), dtype=np.float32))
        assert np.all(data['label'] == np.ones((4,), dtype=np.float32))

    finally:
        shutil.rmtree(temp_dir)


def test_write_parquet_images(orca_context_fixture):
    sc = orca_context_fixture
    temp_dir = tempfile.mkdtemp()

    def generator():
        dataset_path = os.path.join(resource_path, "cat_dog")
        for root, dirs, files in os.walk(os.path.join(dataset_path, "cats")):
            for name in files:
                image_path = os.path.join(root, name)
                yield {"image": image_path, "label": 1, "id": image_path}

        for root, dirs, files in os.walk(os.path.join(dataset_path, "dogs")):
            for name in files:
                image_path = os.path.join(root, name)
                yield {"image": image_path, "label": 0, "id": image_path}

    schema = {
        "image": SchemaField(feature_type=FeatureType.IMAGE, dtype=DType.FLOAT32, shape=(10,)),
        "label": SchemaField(feature_type=FeatureType.NDARRAY, dtype=DType.FLOAT32, shape=(4,)),
        "id": SchemaField(feature_type=FeatureType.SCALAR, dtype=DType.STRING, shape=())
    }

    try:
        ParquetDataset.write("file://" + temp_dir, generator(), schema)
        data, schema = ParquetDataset._read_as_dict_rdd("file://" + temp_dir)
        data = data.collect()[0]
        image_path = data['id']
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        assert image_bytes == data['image']

    finally:
        shutil.rmtree(temp_dir)


def _images_to_mnist_file(images, filepath):
    assert len(images.shape) == 3
    assert images.dtype == np.uint8

    with open(filepath, "wb") as f:
        f.write(int(2051).to_bytes(4, "big"))
        f.write(np.array(images.shape).astype(np.int32).byteswap().tobytes())
        f.write(images.tobytes())


def _labels_to_mnist_file(labels, filepath):
    assert len(labels.shape) == 1
    assert labels.dtype == np.uint8

    with open(filepath, "wb") as f:
        f.write(int(2049).to_bytes(4, "big"))

        f.write(np.array(labels.shape).astype(np.int32).byteswap().tobytes())
        f.write(labels.tobytes())


def test_write_mnist(orca_context_fixture, use_api=False):
    sc = orca_context_fixture
    temp_dir = tempfile.mkdtemp()

    try:
        train_image_file = os.path.join(temp_dir, "train-images")
        train_label_file = os.path.join(temp_dir, "train-labels")
        output_path = os.path.join(temp_dir, "output_dataset")

        images = np.array([[i] * 16 for i in range(20)]).reshape((20, 4, 4)).astype(np.uint8)
        labels = np.array(list(range(20))).reshape((20,)).astype(np.uint8)

        _images_to_mnist_file(images, train_image_file)
        _labels_to_mnist_file(labels, train_label_file)
        if use_api:
            write_parquet("mnist", "file://" + output_path,
                          image_file=train_image_file,
                          label_file=train_label_file)
        else:
            write_mnist(image_file=train_image_file,
                        label_file=train_label_file,
                        output_path="file://" + output_path)
        data, schema = ParquetDataset._read_as_dict_rdd("file://" + output_path)
        data = data.sortBy(lambda x: x['label']).collect()
        images_load = np.reshape(np.stack([d['image'] for d in data]), (-1, 4, 4))
        labels_load = np.stack([d['label'] for d in data])

        assert np.all(images_load == images)
        assert np.all(labels_load == labels_load)

    finally:
        shutil.rmtree(temp_dir)


def test_write_voc(orca_context_fixture, use_api=False):
    sc = orca_context_fixture
    temp_dir = tempfile.mkdtemp()
    try:
        from bigdl.orca.data import SparkXShards
        dataset_path = os.path.join(resource_path, "VOCdevkit")
        output_path = os.path.join(temp_dir, "output_dataset")
        if use_api:
            write_parquet("voc", "file://" + output_path, voc_root_path=dataset_path,
                          splits_names=[(2007, "trainval")])
        else:
            write_voc(dataset_path, splits_names=[(2007, "trainval")],
                      output_path="file://" + output_path)

        data, schema = ParquetDataset._read_as_dict_rdd("file://" + output_path)
        data = data.collect()[0]
        image_path = data["image_id"]
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        assert image_bytes == data['image']

    finally:
        shutil.rmtree(temp_dir)


def test_train_simple(orca_context_fixture):
    sc = orca_context_fixture
    temp_dir = tempfile.mkdtemp()

    try:
        _write_ndarrays(images=np.random.randn(500, 28, 28, 1).astype(np.float32),
                        labels=np.random.randint(0, 10, (500,)).astype(np.int32),
                        output_path="file://" + temp_dir)
        dataset = ParquetDataset.read_as_tf("file://" + temp_dir)

        def preprocess(data):
            return data['image'], data["label"]

        dataset = dataset.map(preprocess)

        import tensorflow as tf
        model = tf.keras.Sequential(
            [tf.keras.layers.Conv2D(20, kernel_size=(5, 5), strides=(1, 1), activation='tanh',
                                    input_shape=(28, 28, 1), padding='valid'),
             tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
             tf.keras.layers.Conv2D(50, kernel_size=(5, 5), strides=(1, 1), activation='tanh',
                                    padding='valid'),
             tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
             tf.keras.layers.Flatten(),
             tf.keras.layers.Dense(500, activation='tanh'),
             tf.keras.layers.Dense(10, activation='softmax'),
             ]
        )

        model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        est = Estimator.from_keras(keras_model=model)
        est.fit(data=dataset,
                batch_size=100,
                epochs=1)

    finally:
        shutil.rmtree(temp_dir)


def test_write_from_directory(orca_context_fixture, use_api=False):
    sc = orca_context_fixture
    temp_dir = tempfile.mkdtemp()
    try:
        label_map = {"cats": 0, "dogs": 1}
        if use_api:
            write_parquet("image_folder", "file://" + temp_dir,
                          directory=os.path.join(resource_path, "cat_dog"),
                          label_map=label_map)
        else:
            write_from_directory(os.path.join(resource_path, "cat_dog"),
                                 label_map, "file://" + temp_dir)
        train_xshard = ParquetDataset._read_as_xshards("file://" + temp_dir)

        data = train_xshard.collect()[0]
        image_path = data["image_id"][0]

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        assert image_bytes == data['image'][0]

    finally:
        shutil.rmtree(temp_dir)


def test_write_parquet_api(orca_context_fixture):
    test_write_mnist(orca_context_fixture, True)
    test_write_voc(orca_context_fixture, True)
    test_write_from_directory(orca_context_fixture, True)
