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
from __future__ import print_function

from keras.layers import *

np.random.seed(1337)  # for reproducibility
import pytest
from keras.applications import *
from bigdl.keras.converter import *
from keras.applications.music_tagger_crnn import MusicTaggerCRNN

from test.bigdl.test_utils import BigDLTestCase, TestModels
from bigdl.keras.backend import *


class TestApplication(BigDLTestCase):

    def assert_model(self, input_data, kmodel, rtol=1e-5, atol=1e-5):
        bmodel = DefinitionLoader.from_kmodel(kmodel)
        self.compare_model(bmodel, kmodel, input_data, rtol, atol)

    def test_lenet(self):
        K.set_image_dim_ordering("th")
        kmodel, input_data, output_data = TestModels.kmodel_seq_lenet_mnist()
        self.modelTest(input_data, kmodel, dump_weights=True)

    def test_text_classification(self):
        # This example demonstrates the use of Convolution1D for text classification.
        # This example is from Keras
        K.set_image_dim_ordering("th")
        import numpy as np
        np.random.seed(1337)  # for reproducibility

        from keras.preprocessing import sequence
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Activation
        from keras.layers import Embedding
        from keras.layers import Convolution1D
        from keras.datasets import imdb

        # set parameters:
        max_features = 5000
        maxlen = 400
        batch_size = 32
        embedding_dims = 50
        nb_filter = 250
        filter_length = 3
        hidden_dims = 250
        nb_epoch = 1

        print('Loading data...')
        (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
        print(len(X_train), 'train sequences')
        print(len(X_test), 'test sequences')

        print('Pad sequences (samples x time)')
        X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
        X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
        print('X_train shape:', X_train.shape)
        print('X_test shape:', X_test.shape)

        print('Build model...')
        model = Sequential()

        model.add(Embedding(max_features,
                            embedding_dims,
                            input_length=maxlen))  # Exception if specify Dropout dropout=0.2

        # we add a Convolution1D, which will learn nb_filter
        # word group filters of size filter_length:
        model.add(Convolution1D(nb_filter=nb_filter,
                                filter_length=filter_length,
                                border_mode='valid',
                                activation='relu',
                                subsample_length=1))
        # we use max pooling:
        model.add(GlobalMaxPooling1D())
        # model.add(GlobalAveragePooling1D())

        # We add a vanilla hidden layer:
        model.add(Dense(hidden_dims))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))

        # We project onto a single unit output layer, and squash it with a sigmoid:
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model = with_bigdl_backend(model)

        model.fit(X_train, y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=(X_test, y_test),
                  is_distributed=True)
        # 2017-09-22 15:53:45 INFO  DistriOptimizer$:657
        # - Top1Accuracy is Accuracy(correct: 21557, count: 25000, accuracy: 0.86228)
        # this result is from GlobalAveragePooling not GlobalMaxPooling.
        model.predict(X_test, is_distributed=True)
        model.evaluate(X_test, y_test, is_distributed=True)
        print(model)

    def test_resnet50(self):
        keras.backend.set_image_dim_ordering("th")
        kmodel = resnet50.ResNet50(include_top=False,
                                   input_shape=(3, 224, 224),
                                   weights=None)
        input_data = np.random.random([2, 3, 224, 224])
        self.assert_model(input_data, kmodel)

    def test_vgg16(self):
        keras.backend.set_image_dim_ordering("th")
        kmodel = vgg16.VGG16(include_top=False,
                             input_shape=(3, 224, 224),
                             weights=None)
        input_data = np.random.random([2, 3, 224, 224])
        self.assert_model(input_data, kmodel)

    def test_vgg19(self):
        keras.backend.set_image_dim_ordering("th")
        kmodel = vgg19.VGG19(include_top=False,
                             input_shape=(3, 224, 224),
                             weights=None)
        input_data = np.random.random([2, 3, 224, 224])
        self.assert_model(input_data, kmodel)

    @pytest.mark.skip(reason="need to fix todo before running the test")
    def test_music_tagger_crnn(self):
        # TODO: For the first BatchNormalization layer in the model, we don't support `axis=3`
        keras.backend.set_image_dim_ordering("th")
        kmodel = MusicTaggerCRNN(include_top=False, weights=None)
        input_data = np.random.random([2, 1, 96, 1366])

        bmodel = DefinitionLoader.from_kmodel(kmodel)
        WeightLoader.load_weights_from_kmodel(bmodel, kmodel)

        keras_output = kmodel.predict(input_data)
        bmodel.training(is_training=False)
        bigdl_output = bmodel.forward(input_data)

        self.assert_allclose(keras_output, bigdl_output, rtol=1e-6, atol=1e-6)

    def test_inception_v3(self):
        keras.backend.set_image_dim_ordering("th")
        kmodel = inception_v3.InceptionV3(include_top=False,
                                          input_shape=(3, 299, 299),
                                          weights=None)
        input_data = np.random.random([2, 3, 299, 299])

        bmodel = DefinitionLoader.from_kmodel(kmodel)
        WeightLoader.load_weights_from_kmodel(bmodel, kmodel)

        keras_output = kmodel.predict(input_data)
        bmodel.training(is_training=False)
        bigdl_output = bmodel.forward(input_data)

        self.assert_allclose(keras_output, bigdl_output, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
