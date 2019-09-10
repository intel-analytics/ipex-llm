#
# Copyright 2018 Analytics Zoo Authors.
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
import pytest

from zoo.feature.common import ChainedPreprocessing, FeatureSet
from zoo.feature.image import *
from test.zoo.pipeline.utils.test_utils import ZooTestCase
import tensorflow as tf
import numpy as np
import os

from zoo.tfpark import KerasModel, TFDataset

resource_path = os.path.join(os.path.split(__file__)[0], "../resources")


class TestTFDataset(ZooTestCase):

    def setup_method(self, method):
        tf.keras.backend.clear_session()
        super(TestTFDataset, self).setup_method(method)

    def create_model(self):
        data = tf.keras.layers.Input(shape=[10])

        x = tf.keras.layers.Flatten()(data)
        x = tf.keras.layers.Dense(10, activation='relu')(x)
        predictions = tf.keras.layers.Dense(2, activation='softmax')(x)

        model = tf.keras.models.Model(inputs=data, outputs=predictions)
        model.compile(optimizer='rmsprop',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def create_training_dataset(self):
        np.random.seed(20)
        x = np.random.rand(20, 10)
        y = np.random.randint(0, 2, (20))

        rdd_x = self.sc.parallelize(x)
        rdd_y = self.sc.parallelize(y)

        rdd = rdd_x.zip(rdd_y)

        dataset = TFDataset.from_rdd(rdd,
                                     features=(tf.float32, [10]),
                                     labels=(tf.int32, []),
                                     batch_size=4,
                                     val_rdd=rdd
                                     )
        return dataset

    def test_dataset_without_batch(self):
        x = np.random.rand(20, 10)
        y = np.random.randint(0, 2, (20))

        rdd_x = self.sc.parallelize(x)
        rdd_y = self.sc.parallelize(y)

        rdd = rdd_x.zip(rdd_y)

        dataset = TFDataset.from_rdd(rdd,
                                     features=(tf.float32, [10]),
                                     labels=(tf.int32, []),
                                     names=["features", "labels"],
                                     val_rdd=rdd
                                     )

        keras_model = self.create_model()
        model = KerasModel(keras_model)
        self.intercept(lambda: model.fit(dataset),
                       "The batch_size of TFDataset must be" +
                       " specified when used in KerasModel fit.")

        dataset = TFDataset.from_rdd(rdd,
                                     features=(tf.float32, [10]),
                                     labels=(tf.int32, []),
                                     names=["features", "labels"],
                                     )
        self.intercept(lambda: model.evaluate(dataset),
                       "The batch_per_thread of TFDataset must be " +
                       "specified when used in KerasModel evaluate.")

        dataset = TFDataset.from_rdd(rdd_x,
                                     features=(tf.float32, [10]),
                                     names=["features", "labels"],
                                     )
        self.intercept(lambda: model.predict(dataset),
                       "The batch_per_thread of TFDataset must be" +
                       " specified when used in KerasModel predict.")

    def create_image_model(self):

        data = tf.keras.layers.Input(shape=[224, 224, 3])
        x = tf.keras.layers.Flatten()(data)
        predictions = tf.keras.layers.Dense(10, activation='softmax')(x)

        model = tf.keras.models.Model(inputs=data, outputs=predictions)
        model.compile(optimizer='rmsprop',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return KerasModel(model)

    def create_image_set(self, with_label):
        image_set = self.get_raw_image_set(with_label)
        transformer = ChainedPreprocessing([ImageResize(256, 256),
                                            ImageRandomCrop(224, 224, True),
                                            ImageMatToTensor(format="NHWC"),
                                            ImageSetToSample(input_keys=["imageTensor"],
                                                             target_keys=["label"]
                                                             if with_label else None)])
        image_set = image_set.transform(transformer)
        return image_set

    def create_train_features_Set(self):
        image_set = self.get_raw_image_set(with_label=True)
        feature_set = FeatureSet.image_frame(image_set.to_image_frame())
        train_transformer = ChainedPreprocessing([ImageBytesToMat(),
                                                  ImageResize(256, 256),
                                                  ImageRandomCrop(224, 224),
                                                  ImageRandomPreprocessing(ImageHFlip(), 0.5),
                                                  ImageChannelNormalize(
                                                      0.485, 0.456, 0.406,
                                                      0.229, 0.224, 0.225),
                                                  ImageMatToTensor(to_RGB=True, format="NHWC"),
                                                  ImageSetToSample(input_keys=["imageTensor"],
                                                                   target_keys=["label"])
                                                  ])
        feature_set = feature_set.transform(train_transformer)
        return feature_set

    def test_training_for_imageset(self):

        model = self.create_image_model()
        image_set = self.create_image_set(with_label=True)
        training_dataset = TFDataset.from_image_set(image_set,
                                                    image=(tf.float32, [224, 224, 3]),
                                                    label=(tf.int32, [1]),
                                                    batch_size=4)
        model.fit(training_dataset)

    def test_training_for_feature_set(self):
        model = self.create_image_model()
        feature_set = self.create_train_features_Set()
        training_dataset = TFDataset.from_feature_set(feature_set,
                                                      features=(tf.float32, [224, 224, 3]),
                                                      labels=(tf.int32, [1]),
                                                      batch_size=8)
        model.fit(training_dataset)

    def test_evaluation_for_imageset(self):

        model = self.create_image_model()
        image_set = self.create_image_set(with_label=True)
        eval_dataset = TFDataset.from_image_set(image_set,
                                                image=(tf.float32, [224, 224, 3]),
                                                label=(tf.int32, [1]),
                                                batch_per_thread=1)

        model.evaluate(eval_dataset)

    def test_predict_for_imageset(self):
        model = self.create_image_model()
        image_set = self.create_image_set(with_label=False)

        predict_dataset = TFDataset.from_image_set(image_set,
                                                   image=(tf.float32, [224, 224, 3]),
                                                   batch_per_thread=1)
        results = model.predict(predict_dataset).get_predict().collect()
        assert all(r[1] is not None for r in results)

    def test_gradient_clipping(self):

        data = tf.keras.layers.Input(shape=[10])

        x = tf.keras.layers.Flatten()(data)
        x = tf.keras.layers.Dense(10, activation='relu')(x)
        predictions = tf.keras.layers.Dense(2, activation='softmax')(x)

        model = tf.keras.models.Model(inputs=data, outputs=predictions)
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=1, clipvalue=1e-8),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model = KerasModel(model)

        pre_weights = model.get_weights()

        dataset = self.create_training_dataset()

        # 5 iterations
        model.fit(dataset)

        current_weight = model.get_weights()

        np.all(np.abs((current_weight[0] - pre_weights[0])) < 1e-7)

    def test_tf_dataset_with_list_feature(self):
        np.random.seed(20)
        x = np.random.rand(20, 10)
        y = np.random.randint(0, 2, (20))

        rdd_x = self.sc.parallelize(x)
        rdd_y = self.sc.parallelize(y)

        rdd = rdd_x.zip(rdd_y)

        dataset = TFDataset.from_rdd(rdd,
                                     features=[(tf.float32, [10]), (tf.float32, [10])],
                                     labels=(tf.int32, []),
                                     batch_size=4,
                                     val_rdd=rdd
                                     )

        for idx, tensor in enumerate(dataset.feature_tensors):
            assert tensor.name == "list_input_" + str(idx) + ":0"

    def test_tfdataset_with_tfrecord(self):
        model = tf.keras.Sequential(
            [tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
             tf.keras.layers.Dense(10, activation='softmax'),
             ]
        )

        model.compile(optimizer='rmsprop',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        keras_model = KerasModel(model)

        def parse_fn(example):
            keys_to_features = {
                'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
                'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
                'image/class/label': tf.FixedLenFeature(
                    [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
            }

            items_to_handlers = {
                'image': tf.contrib.slim.tfexample_decoder.Image(shape=[28, 28, 1], channels=1),
                'label': tf.contrib.slim.tfexample_decoder.Tensor('image/class/label', shape=[]),
            }

            decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
                keys_to_features, items_to_handlers)
            results = decoder.decode(example)

            if len(results[0].shape) > 0:
                feature = results[0]
                label = results[1]
            else:
                feature = results[1]
                label = results[0]

            return feature, label

        train_path = os.path.join(resource_path, "tfrecord/mnist_train.tfrecord")
        test_path = os.path.join(resource_path, "tfrecord/mnist_test.tfrecord")
        dataset = TFDataset.from_tfrecord(train_path,
                                          parse_fn=parse_fn, batch_size=8,
                                          validation_file_path=test_path)

        keras_model.fit(dataset)


if __name__ == "__main__":
    pytest.main([__file__])
