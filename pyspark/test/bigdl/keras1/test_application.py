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
from bigdl.keras1.backend import use_bigdl_backend
import numpy as np
import pytest

from test.bigdl.test_utils import BigDLTestCase, TestModels


class TestApplication(BigDLTestCase):

    def test_lenet(self):
        kmodel, input_data, output_data = TestModels.kmodel_seq_lenet_mnist()
        self.modelTest(input_data, kmodel, dump_weights=True)

    def test_text_classification(self):
        '''This example demonstrates the use of Convolution1D for text classification.
           This example is from Keras
        '''

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
        nb_epoch = 2

        print('Loading data...')
        (X_train, y_train), (X_test, y_test) = imdb.load_data(
            path="/Users/lizhichao/god/data/imdb_full.pkl", nb_words=max_features)
        print(len(X_train), 'train sequences')
        print(len(X_test), 'test sequences')

        print('Pad sequences (samples x time)')
        X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
        X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
        print('X_train shape:', X_train.shape)
        print('X_test shape:', X_test.shape)

        print('Build model...')
        model = Sequential()

        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        # model.add(Embedding(max_features,
        #                     embedding_dims,
        #                     input_length=maxlen,
        #                     dropout=0.2)) # Exception if specify Dropout

        model.add(Embedding(max_features,
                            embedding_dims,
                            input_length=maxlen))  # Exception if specify Dropout

        # we add a Convolution1D, which will learn nb_filter
        # word group filters of size filter_length:
        model.add(Convolution1D(nb_filter=nb_filter,
                                filter_length=filter_length,
                                border_mode='valid',
                                activation='relu',
                                subsample_length=1))
        # we use max pooling:
        # model.add(GlobalMaxPooling1D()) #TODO: Why there's exception for GlobalMaxPooling1D?
        model.add(GlobalAveragePooling1D())

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
        model = use_bigdl_backend(model)

        # model.fit(X_train, y_train,
        #           batch_size=batch_size,
        #           nb_epoch=nb_epoch,
        #           validation_data=(X_test, y_test))
        # 2017-09-22 15:53:45 INFO  DistriOptimizer$:657
        # - Top1Accuracy is Accuracy(correct: 21557, count: 25000, accuracy: 0.86228)
        # this result is from GlobalAveragePooling not GlobalMaxPooling.
        model.predict(X_test)  # OK
        model.evaluate(X_test, y_test)
        print(model)

if __name__ == "__main__":
    pytest.main([__file__])
