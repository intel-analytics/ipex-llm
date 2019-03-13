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

# Source url of dataset is from boston_housing.py from Keras Open Source Project
#
# The MIT License (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np

from bigdl.dataset import base


def load_data(path='boston_housing.npz', dest_dir='/tmp/.zoo/dataset', test_split=0.2):
    """Loads the Boston Housing dataset, the source url of download
       is copied from keras.datasets
    # Arguments
        dest_dir: where to cache the data (relative to `~/.zoo/dataset`).
        nb_words: number of words to keep, the words are already indexed by frequency
                  so that the less frequent words would be abandoned
        oov_char: index to pad the abandoned words, if None, one abandoned word
                  would be taken place with its next word and total length -= 1
        test_split: the ratio to split part of dataset to test data,
                    the remained data would be train data

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """

    path = base.maybe_download(path,
                               dest_dir,
                               'https://s3.amazonaws.com/keras-datasets/boston_housing.npz')

    with np.load(path) as f:
        x = f['x']
        y = f['y']

    shuffle_by_seed([x, y])
    split_index = int(len(x) * (1 - test_split))

    x_train, y_train = x[:split_index], y[:split_index]

    x_test, y_test = x[split_index:], y[split_index:]

    return (x_train, y_train), (x_test, y_test)


def shuffle_by_seed(arr_list, seed=0):
    for arr in arr_list:
        np.random.seed(seed)
        np.random.shuffle(arr)
