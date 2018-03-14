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
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import *
from bigdl.dataset import mnist


def build_model(class_num):
    model = Sequential()
    model.add(Reshape((1, 28, 28), input_shape=(28, 28, 1)))
    model.add(Convolution2D(32, 3, 3, activation="relu"))
    model.add(Convolution2D(32, 3, 3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(class_num, activation="softmax"))
    return model


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-b", "--batchSize", type=int, dest="batchSize", default="128")
    parser.add_option("-n", "--maxEpoch", type=int, dest="maxEpoch", default="20")
    parser.add_option("-d", "--dataPath", dest="dataPath", default="/tmp/mnist")
    (options, args) = parser.parse_args(sys.argv)

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data(options.dataPath)

    model = build_model(10)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=options.batchSize, nb_epoch=options.maxEpoch,
              validation_data=(X_test, Y_test))
