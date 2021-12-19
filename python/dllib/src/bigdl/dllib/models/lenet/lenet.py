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

from optparse import OptionParser
import bigdl.dllib.keras.Sequential
from bigdl.dllib.keras.layers import *
from bigdl.dllib.feature.dataset import mnist


def build_model(class_num):
    model = Sequential()
    model.add(Reshape((1, 28, 28), input_shape=(28, 28, 1)))
    model.add(Convolution2D(6, 5, 5, activation="tanh", name="conv1_5x5"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(12, 5, 5, activation="tanh", name="conv2_5x5"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(100, activation="tanh", name="fc1"))
    model.add(Dense(class_num, activation="softmax", name="fc2"))
    return model


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-b", "--batchSize", type=int, dest="batchSize", default="128")
    parser.add_option("-n", "--maxEpoch", type=int, dest="maxEpoch", default="20")
    parser.add_option("-d", "--dataPath", dest="dataPath", default="/tmp/mnist")
    (options, args) = parser.parse_args(sys.argv)

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data(options.dataPath)
    Y_train = Y_train - 1
    Y_test = Y_test - 1

    model = build_model(10)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=options.batchSize, nb_epoch=options.maxEpoch,
              validation_data=(X_test, Y_test))
