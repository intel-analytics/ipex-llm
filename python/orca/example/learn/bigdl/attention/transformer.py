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


import argparse
import numpy as np
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.preprocessing import sequence

from zoo.pipeline.api.keras.models import Model
from zoo.pipeline.api.keras.layers import *
from zoo.pipeline.api.keras.objectives import SparseCategoricalCrossEntropy
from zoo.orca.data import XShards
from zoo.orca.learn.metrics import Accuracy
from zoo.orca.learn.bigdl.estimator import Estimator
from zoo.orca import init_orca_context, stop_orca_context

parser = argparse.ArgumentParser()
parser.add_argument('--cluster_mode', type=str, default="local",
                    help='The mode for the Spark cluster. local or yarn.')
args = parser.parse_args()
cluster_mode = args.cluster_mode
conf = {"spark.executor.extraJavaOptions": "-Xss512m",
        "spark.driver.extraJavaOptions": "-Xss512m"}
max_features = 20000
max_len = 200

if cluster_mode == "local":
    sc = init_orca_context(cluster_mode="local", cores=8,
                           memory="100g",
                           driver_memory="20g",
                           conf=conf
                           )
elif cluster_mode == "yarn":
    sc = init_orca_context(cluster_mode="yarn-client", num_nodes=8, cores=8,
                           memory="100g",
                           driver_memory="20g",
                           conf=conf
                           )
else:
    print("init_orca_context failed. cluster_mode should be either 'local' or 'yarn' but got "
          + cluster_mode)

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

train_pos = np.zeros((len(x_train), max_len), dtype=np.int32)
val_pos = np.zeros((len(x_test), max_len), dtype=np.int32)
for i in range(0, len(x_train)):
    train_pos[i, :] = np.arange(max_len)
    val_pos[i, :] = np.arange(max_len)

train_dataset = XShards.partition({"x": (x_train, train_pos), "y": np.array(y_train)})
val_dataset = XShards.partition({"x": (x_test, val_pos), "y": np.array(y_test)})

token_shape = (max_len,)
position_shape = (max_len,)
token_input = Input(shape=token_shape)
position_input = Input(shape=position_shape)
O_seq = TransformerLayer.init(
    vocab=max_features, hidden_size=128, n_head=8, seq_len=max_len)([token_input, position_input])
# Select the first output of the Transformer. The second is the pooled output.
O_seq = SelectTable(0)(O_seq)
O_seq = GlobalAveragePooling1D()(O_seq)
O_seq = Dropout(0.2)(O_seq)
outputs = Dense(2, activation='softmax')(O_seq)

model = Model([token_input, position_input], outputs)
model.summary()
batch_size = 128
print('Train...')

est = Estimator.from_bigdl(model=model, loss=SparseCategoricalCrossEntropy(),
                           optimizer=Adam(), metrics=[Accuracy()])
est.fit(data=train_dataset,
        batch_size=batch_size,
        epochs=1)
print("Train finished.")

print('Evaluating...')
result = est.evaluate(val_dataset)
print(result)

print("finished...")
stop_orca_context()
