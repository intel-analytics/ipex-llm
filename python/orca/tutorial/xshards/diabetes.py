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
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras

import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import bigdl.orca.data.pandas
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.tf2.estimator import Estimator

init_orca_context(cluster_mode="local", cores=4, memory="3g")

path = 'pima-indians-diabetes.csv'
data_shard = bigdl.orca.data.pandas.read_csv(path, header=None)
column = list(data_shard.get_schema()['columns'])


def model_creator(config):
        model = Sequential()
        model.add(Dense(12, input_shape=(8,), activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

data_shard = data_shard.assembleFeatureLabelCols(featureCols=column[:-1],
                                                 labelCols=list(column[-1]))

batch_size = 16
train_steps = math.ceil(len(data_shard) / batch_size)
est = Estimator.from_keras(model_creator=model_creator)

est.fit(data=data_shard,
        batch_size=batch_size,
        epochs=150,
        steps_per_epoch=train_steps)
stop_orca_context()
