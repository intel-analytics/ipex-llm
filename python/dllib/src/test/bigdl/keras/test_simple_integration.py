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
import shutil

from zoo.feature.common import ChainedPreprocessing
from zoo.feature.image import *
from zoo.pipeline.api.keras.layers import *
from zoo.pipeline.api.keras.models import *
from test.zoo.pipeline.utils.test_utils import ZooTestCase

np.random.seed(1337)  # for reproducibility


class TestSimpleIntegration(ZooTestCase):

    def test_sequential(self):
        model = Sequential()
        model.add(InputLayer(input_shape=(8, )))
        model.add(Dense(10))
        model.add(Dense(12))
        input_shape = model.get_input_shape()
        output_shape = model.get_output_shape()
        np.testing.assert_allclose((8,), input_shape[1:])
        np.testing.assert_allclose((12,), output_shape[1:])

    def test_graph(self):
        x1 = Input(shape=(8, ))
        x2 = Input(shape=(6, ))
        y1 = Dense(10)(x1)
        y2 = Dense(10)(x2)
        model = Model([x1, x2], [y1, y2])
        tmp_log_dir = create_tmp_path()
        model.save_graph_topology(tmp_log_dir)
        input_shapes = model.get_input_shape()
        output_shapes = model.get_output_shape()
        np.testing.assert_allclose((8, ), input_shapes[0][1:])
        np.testing.assert_allclose((6, ), input_shapes[1][1:])
        np.testing.assert_allclose((10, ), output_shapes[0][1:])
        np.testing.assert_allclose((10, ), output_shapes[1][1:])
        shutil.rmtree(tmp_log_dir)

    def test_training_with_tensorboard_checkpoint_gradientclipping(self):
        model = Sequential()
        model.add(Dense(8, input_shape=(32, 32, )))
        model.add(Flatten())
        model.add(Dense(4, activation="softmax"))
        X_train = np.random.random([200, 32, 32])
        y_train = np.random.randint(4, size=(200, ))
        X_test = np.random.random([40, 32, 32])
        y_test = np.random.randint(4, size=(40, ))
        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=['accuracy'])
        tmp_log_dir = create_tmp_path()
        tmp_checkpoint_path = create_tmp_path()
        os.mkdir(tmp_checkpoint_path)
        model.set_tensorboard(tmp_log_dir, "training_test")
        model.set_checkpoint(tmp_checkpoint_path)
        model.set_constant_gradient_clipping(0.01, 0.03)
        model.fit(X_train, y_train, batch_size=112, nb_epoch=2, validation_data=(X_test, y_test))
        model.clear_gradient_clipping()
        model.fit(X_train, y_train, batch_size=112, nb_epoch=2, validation_data=(X_test, y_test))
        model.set_gradient_clipping_by_l2_norm(0.2)
        model.fit(X_train, y_train, batch_size=112, nb_epoch=2, validation_data=(X_test, y_test))
        model.evaluate(X_test, y_test, batch_size=112)
        model.predict(X_test)
        model.predict_classes(X_test)
        shutil.rmtree(tmp_log_dir)
        shutil.rmtree(tmp_checkpoint_path)

    def test_training_without_validation(self):
        model = Sequential()
        model.add(Dense(4, activation="relu", input_shape=(10, )))
        x = np.random.random([300, 10])
        y = np.random.random([300, ])
        model.compile(optimizer="sgd", loss="mae")
        model.fit(x, y, batch_size=112, nb_epoch=2)
        model.predict(x)

    def test_training_imageset(self):
        images = []
        labels = []
        for i in range(0, 32):
            features = np.random.uniform(0, 1, (200, 200, 3))
            label = np.array([2])
            images.append(features)
            labels.append(label)
        image_set = DistributedImageSet(self.sc.parallelize(images),
                                        self.sc.parallelize(labels))

        transformer = ChainedPreprocessing(
            [ImageBytesToMat(), ImageResize(256, 256), ImageCenterCrop(224, 224),
             ImageChannelNormalize(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
             ImageMatToTensor(), ImageSetToSample(target_keys=['label'])])
        data_rdd = image_set.transform(transformer)

        model = Sequential()
        model.add(Convolution2D(1, 5, 5, input_shape=(3, 224, 224)))
        model.add(Reshape((1*220*220, )))
        model.add(Dense(20, activation="softmax"))
        model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(data_rdd, batch_size=8, nb_epoch=2, validation_data=data_rdd)

    def test_remove_batch(self):
        from zoo.pipeline.api.utils import remove_batch
        assert remove_batch([2, 3, 4]) == [3, 4]
        assert remove_batch([[2, 6, 7], [2, 3, 4]]) == [[6, 7], [3, 4]]

    def test_sequential_to_model(self):
        seq = Sequential()
        seq.add(Dense(8, input_shape=(32, 32, )))
        seq.add(Flatten())
        seq.add(Dense(4, activation="softmax"))
        seq.to_model()

    def test_keras_net_layers(self):
        x1 = Input(shape=(8, ))
        x2 = Input(shape=(6, ))
        y1 = Dense(10)(x1)
        y2 = Dense(10)(x2)
        model = Model([x1, x2], [y1, y2])
        assert len(model.layers) == 4

    def test_keras_net_flatten_layers(self):
        x1 = Input(shape=(8, ))
        x2 = Input(shape=(6, ))
        y1 = Dense(10)(x1)
        y2 = Dense(10)(x2)
        model = Model([x1, x2], [y1, y2])
        assert len(model.flattened_layers()) == 4

    def test_keras_get_layer(self):
        x1 = Input(shape=(8,))
        y1 = Dense(10, name="Dense")(x1)
        model = Model([x1], [y1])
        layer = model.get_layer("Dense")
        assert layer.name() == "Dense"

    def test_create_image_config(self):
        from zoo.models.image.common.image_config import ImageConfigure
        from zoo.feature.image.imagePreprocessing import ImageResize
        from zoo.feature.common import ChainedPreprocessing
        ImageConfigure(
            pre_processor=ImageResize(224, 224))
        ImageConfigure(
            pre_processor=ChainedPreprocessing([ImageResize(224, 224), ImageResize(224, 224)]))

    def test_model_summary_sequential(self):
        model = Sequential()
        model.add(LSTM(input_shape=(16, 32), output_dim=8, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(15, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(output_dim=1))
        model.summary()

    def test_model_summary_graph(self):
        x = Input(shape=(8, ))
        y = Dense(10)(x)
        z = Dense(12)(y)
        model = Model(x, z)
        model.summary()


if __name__ == "__main__":
    pytest.main([__file__])
