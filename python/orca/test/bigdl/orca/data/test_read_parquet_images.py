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

import pytest
from unittest import TestCase
import os
from bigdl.orca.data.image.parquet_dataset import ParquetDataset, read_parquet
from bigdl.orca.data.image.utils import DType, FeatureType, SchemaField
import tensorflow as tf

from bigdl.orca.ray import RayContext

resource_path = os.path.join(os.path.split(__file__)[0], "../resources")
WIDTH, HEIGHT, NUM_CHANNELS = 224, 224, 3


def images_generator():
    dataset_path = os.path.join(resource_path, "cat_dog")
    for root, dirs, files in os.walk(os.path.join(dataset_path, "cats")):
        for name in files:
            image_path = os.path.join(root, name)
            yield {"image": image_path, "label": 1, "id": image_path}

    for root, dirs, files in os.walk(os.path.join(dataset_path, "dogs")):
        for name in files:
            image_path = os.path.join(root, name)
            yield {"image": image_path, "label": 0, "id": image_path}


images_schema = {
    "image": SchemaField(feature_type=FeatureType.IMAGE, dtype=DType.FLOAT32, shape=()),
    "label": SchemaField(feature_type=FeatureType.SCALAR, dtype=DType.FLOAT32, shape=()),
    "id": SchemaField(feature_type=FeatureType.SCALAR, dtype=DType.STRING, shape=())
}


def parse_data_train(image, label):
    image = tf.io.decode_jpeg(image, NUM_CHANNELS)
    image = tf.image.resize(image, size=(WIDTH, HEIGHT))
    image = tf.reshape(image, [WIDTH, HEIGHT, NUM_CHANNELS])
    return image, label


def model_creator(config):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


class TestReadParquet(TestCase):
    def test_read_parquet_images_tf_dataset(self):
        temp_dir = tempfile.mkdtemp()

        try:
            ParquetDataset.write("file://" + temp_dir, images_generator(),
                                 images_schema, block_size=4)
            path = "file://" + temp_dir
            output_types = {"id": tf.string, "image": tf.string, "label": tf.float32}
            dataset = read_parquet("tf_dataset", path=path, output_types=output_types)
            for dt in dataset.take(1):
                print(dt.keys())

            num_shards, rank = 3, 1
            dataset_shard = read_parquet("tf_dataset", path=path, config={"num_shards": num_shards,
                                                                          "rank": rank},
                                         output_types=output_types)
            assert len(list(dataset_shard)) <= len(list(dataset)) // num_shards, \
                "len of dataset_shard should be 1/`num_shards` of the whole dataset."

            dataloader = read_parquet("dataloader", path=path)
            dataloader_shard = read_parquet("dataloader", path=path,
                                            config={"num_shards": num_shards, "rank": rank})
            cur_dl = iter(dataloader_shard)
            cur_count = 0
            while True:
                try:
                    print(next(cur_dl)['label'])
                    cur_count += 1
                except StopIteration:
                    break
            assert cur_count == len(list(dataset_shard))
        finally:
            shutil.rmtree(temp_dir)

    def test_parquet_images_training(self):
        from bigdl.orca.learn.tf2 import Estimator
        temp_dir = tempfile.mkdtemp()
        try:
            ParquetDataset.write("file://" + temp_dir, images_generator(), images_schema)
            path = "file://" + temp_dir
            output_types = {"id": tf.string, "image": tf.string, "label": tf.float32}
            output_shapes = {"id": (), "image": (), "label": ()}

            def data_creator(config, batch_size):
                dataset = read_parquet("tf_dataset", path=path,
                                       output_types=output_types, output_shapes=output_shapes)
                dataset = dataset.shuffle(10)
                dataset = dataset.map(lambda data_dict:
                                      (data_dict["image"], data_dict["label"]))
                dataset = dataset.map(parse_data_train)
                dataset = dataset.batch(batch_size)
                return dataset

            ray_ctx = RayContext.get()
            trainer = Estimator.from_keras(model_creator=model_creator)
            trainer.fit(data=data_creator,
                        epochs=1,
                        batch_size=2)
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__])
