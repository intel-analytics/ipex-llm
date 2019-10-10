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

import tensorflow as tf
from zoo import init_nncontext
from zoo.tfpark import TFDataset, TFPredictor
import numpy as np
import sys
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

from bigdl.dataset import mnist
from bigdl.dataset.transformer import *

DISTRIBUTED = True


def main(data_num):

    data = Input(shape=[28, 28, 1])

    x = Flatten()(data)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(inputs=data, outputs=predictions)

    model.load_weights("/tmp/mnist_keras.h5")

    if DISTRIBUTED:
        # using RDD api to do distributed evaluation
        sc = init_nncontext()
        # get data, pre-process and create TFDataset
        (images_data, labels_data) = mnist.read_data_sets("/tmp/mnist", "test")
        image_rdd = sc.parallelize(images_data[:data_num])
        labels_rdd = sc.parallelize(labels_data[:data_num])
        rdd = image_rdd.zip(labels_rdd) \
            .map(lambda rec_tuple: [normalizer(rec_tuple[0], mnist.TRAIN_MEAN, mnist.TRAIN_STD)])

        dataset = TFDataset.from_rdd(rdd,
                                     names=["features"],
                                     shapes=[[28, 28, 1]],
                                     types=[tf.float32],
                                     batch_per_thread=20
                                     )
        predictor = TFPredictor.from_keras(model, dataset)

        accuracy = predictor.predict().zip(labels_rdd).map(lambda x: np.argmax(x[0]) == x[1]).mean()

        print("predict accuracy is %s" % accuracy)

    else:
        # using keras api for local evaluation
        model.compile(optimizer='rmsprop',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        (images_data, labels_data) = mnist.read_data_sets("/tmp/mnist", "test")
        images_data = normalizer(images_data, mnist.TRAIN_MEAN, mnist.TRAIN_STD)
        result = model.evaluate(images_data, labels_data)
        print(model.metrics_names)
        print(result)


if __name__ == '__main__':

    data_num = 10000

    if len(sys.argv) > 1:
        data_num = int(sys.argv[1])
    main(data_num)
