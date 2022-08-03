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
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import bigdl.orca.data.pandas
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.tf.estimator import Estimator
from bigdl.orca.data.utils import *

init_orca_context(cluster_mode="local", cores=4, memory="3g")

path = 'pima-indians-diabetes-test.csv'
data_shard = bigdl.orca.data.pandas.read_csv(path)

model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

data_shard = assembleFeatureLabelCols(data_shard,
                                      featureCols=['f1', 'f2', 'f3',
                                                   'f4', 'f5', 'f6', 'f7', 'f8'],
                                      labelCols=['label'])

est = Estimator.from_keras(keras_model=model)
est.fit(data=data_shard,
        batch_size=16,
        epochs=150)
