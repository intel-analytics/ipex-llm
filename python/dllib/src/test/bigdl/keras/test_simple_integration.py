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

    def test_training_with_tensorboard_checkpoint(self):
        model = Sequential()
        model.add(Dense(8, input_shape=(32, 32, )))
        model.add(Flatten())
        model.add(Dense(4, activation="softmax"))
        X_train = np.random.random([200, 32, 32])
        y_train = np.random.randint(4, size=(200, )) + 1
        X_test = np.random.random([40, 32, 32])
        y_test = np.random.randint(4, size=(40, )) + 1
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
        tmp_log_dir = create_tmp_path()
        tmp_checkpoint_path = create_tmp_path()
        os.mkdir(tmp_checkpoint_path)
        model.set_tensorboard(tmp_log_dir, "training_test")
        model.set_checkpoint(tmp_checkpoint_path)
        model.fit(X_train, y_train, batch_size=112, nb_epoch=2, validation_data=(X_test, y_test))
        model.evaluate(X_test, y_test, batch_size=112)
        model.predict(X_test)
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

    def test_training_imagefeature_dataset(self):
        images = []
        labels = []
        for i in range(0, 8):
            features = np.random.uniform(0, 1, (200, 200, 3))
            label = np.array([2])
            images.append(features)
            labels.append(label)
        image_frame = DistributedImageFrame(self.sc.parallelize(images),
                                            self.sc.parallelize(labels))

        transformer = Pipeline([BytesToMat(), Resize(256, 256), CenterCrop(224, 224),
                                ChannelNormalize(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
                                MatToTensor(), ImageFrameToSample(target_keys=['label'])])
        data_set = DataSet.image_frame(image_frame).transform(transformer)

        model = Sequential()
        model.add(Convolution2D(1, 5, 5, input_shape=(3, 224, 224)))
        model.add(Reshape((1*220*220, )))
        model.add(Dense(20, activation="softmax"))
        model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(data_set, batch_size=8, nb_epoch=2, validation_data=data_set)


if __name__ == "__main__":
   pytest.main([__file__])
