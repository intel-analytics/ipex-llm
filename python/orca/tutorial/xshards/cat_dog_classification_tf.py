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
# This example is adapted from
# https://www.kaggle.com/code/ruchibahl18/cats-vs-dogs-basic-cnn-tutorial

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
from bigdl.orca.data.shard import SparkXShards

import bigdl.orca.data.image
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.tf2.estimator import Estimator
import numpy as np

sc = init_orca_context(cluster_mode="local", cores=4, memory="4g")

path = '/Users/guoqiong/intelWork/data/dogs-vs-cats/small/'

data_shard = bigdl.orca.data.image.read_images_spark(path)

def get_label(im):
    filename = im['origin']
    label = [1] if 'dog' in filename.split('/')[-1] else [0]
    return {'x': im['pilimage'], 'y': [label]}


def crop(data):
    im = data['x']
    width, height = im.size  # Get dimensions
    left = width / 4
    top = height / 4
    right = 3 * width / 4
    bottom = 3 * height / 4
    cropped = im.crop((left, top, right, bottom))
    return {'x': cropped, 'y': data['y']}


def resize(im):
    size = (80, 80)
    return {'x': im['x'].resize(size), 'y': im['y']}


to_nparray = lambda x: {'x': np.array([np.asarray(x['x'])]), 'y': np.array([x['y']])}


data_shard = data_shard.transform_shard(get_label)
data_shard = data_shard.transform_shard(crop)
data_shard = data_shard.transform_shard(resize)
data_shard = data_shard.transform_shard(to_nparray)


def model_creator(config):
    model = Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(80, 80, 3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Add another:
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    # Add a softmax layer with 10 output units:
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer="adam",
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


est = Estimator.from_keras(model_creator=model_creator)
est.fit(data=data_shard,
        batch_size=8,
        steps_per_epoch=50,
        epochs=2)
stop_orca_context()
