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
# ==============================================================================
#
# Copyright 2018 The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# This example classifies movie reviews as positive or negative using the text of the review,
# and is adapted from
# https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/keras/basic_text_classification.ipynb

import tensorflow as tf

from tensorflow import keras

import argparse

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.tf.estimator import Estimator

parser = argparse.ArgumentParser()
parser.add_argument('--cluster_mode', type=str, default="local",
                    help='The mode for the Spark cluster. local, yarn or spark-submit.')
parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--download', type=bool, default=True, help='download dataset or not')
parser.add_argument('--data_dir', type=str, default="./dataset", help='The path of datesets where includes imdb.npz.')
args = parser.parse_args()
cluster_mode = args.cluster_mode
download = args.download
if cluster_mode == "local":
    init_orca_context(cluster_mode="local", cores=4, memory="3g")
elif cluster_mode.startswith("yarn"):
    init_orca_context(cluster_mode=cluster_mode, num_nodes=2, cores=2, driver_memory="3g")
elif cluster_mode == "spark-submit":
    init_orca_context(cluster_mode="spark-submit")
else:
    print("init_orca_context failed. cluster_mode should be one of 'local', 'yarn' and 'spark-submit' but got "
          + cluster_mode)

print(tf.__version__)

imdb = keras.datasets.imdb

if download == True:
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
else:
    import numpy as np
    import os
    from tensorflow.keras.utils import get_file
    path = os.path.join(args.data_dir, 'imdb.npz')
    num_words=10000
    skip_top=0
    maxlen=None
    seed=113
    start_char=1
    oov_char=2
    index_from=3
    with np.load(path, allow_pickle=True) as f:  # pylint: disable=unexpected-keyword-arg
        x_train, labels_train = f['x_train'], f['y_train']
        x_test, labels_test = f['x_test'], f['y_test']

    rng = np.random.RandomState(seed)
    indices = np.arange(len(x_train))
    rng.shuffle(indices)
    x_train = x_train[indices]
    labels_train = labels_train[indices]

    indices = np.arange(len(x_test))
    rng.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]

    if start_char is not None:
        x_train = [[start_char] + [w + index_from for w in x] for x in x_train]
        x_test = [[start_char] + [w + index_from for w in x] for x in x_test]
    elif index_from:
        x_train = [[w + index_from for w in x] for x in x_train]
        x_test = [[w + index_from for w in x] for x in x_test]

    if maxlen:
        x_train, labels_train = _remove_long_seq(maxlen, x_train, labels_train)
        x_test, labels_test = _remove_long_seq(maxlen, x_test, labels_test)
    if not x_train or not x_test:
        raise ValueError('After filtering for sequences shorter than maxlen='
                        f'{str(maxlen)}, no sequence was kept. Increase maxlen.')

    xs = x_train + x_test
    labels = np.concatenate([labels_train, labels_test])

    if not num_words:
        num_words = max(max(x) for x in xs)

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        xs = [
            [w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs
        ]
    else:
        xs = [[w for w in x if skip_top <= w < num_words] for x in xs]

    idx = len(x_train)
    x_train, y_train = np.array(xs[:idx], dtype='object'), labels[:idx]
    x_test, y_test = np.array(xs[idx:], dtype='object'), labels[idx:]
    (train_data, train_labels), (test_data, test_labels) = (x_train, y_train), (x_test, y_test)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

print(train_data[0])

len(train_data[0]), len(train_data[1])

# A dictionary mapping words to an integer index
if download == True:
    word_index = imdb.get_word_index()
else:
    import json
    path = os.path.join(args.data_dir, 'imdb_word_index.json')
    with open(path) as f:
        word_index = json.load(f)

# The first indices are reserved
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


decode_review(train_data[0])

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

len(train_data[0]), len(train_data[1])

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

train_dataset = tf.data.Dataset.from_tensor_slices((partial_x_train, partial_y_train))
validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
est = Estimator.from_keras(keras_model=model)
est.fit(data=train_dataset,
        batch_size=512,
        epochs=args.epochs,
        validation_data=validation_dataset
        )
results = est.evaluate(validation_dataset)
print(results)
stop_orca_context()
