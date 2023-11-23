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


from bigdl.dllib.feature.dataset import base
import numpy as np

from six.moves import cPickle


def download_imdb(dest_dir):
    """Download pre-processed IMDB movie review data

    :argument
        dest_dir: destination directory to store the data

    :return
        The absolute path of the stored data
    """
    file_name = "imdb_full.pkl"
    file_abs_path = base.maybe_download(file_name,
                                        dest_dir,
                                        'https://s3.amazonaws.com/text-datasets/imdb_full.pkl')
    return file_abs_path


def load_data(dest_dir='/tmp/.bigdl/dataset', nb_words=None, oov_char=2):
    """Load IMDB dataset.

    :argument
        dest_dir: where to cache the data (relative to `~/.bigdl/dataset`).
        nb_words: number of words to keep, the words are already indexed by frequency
                  so that the less frequent words would be abandoned
        oov_char: index to pad the abandoned words, if None, one abandoned word
                  would be taken place with its next word and total length -= 1
    :return
        the train, test separated IMDB dataset.
    """
    path = download_imdb(dest_dir)

    with open(path, 'rb') as f:
        (x_train, y_train), (x_test, y_test) = cPickle.load(f)
        # imdb.pkl would return different numbers of variables, not 4

    shuffle_by_seed([x_train, y_train, x_test, y_test])
    x = x_train + x_test

    if not nb_words:
        nb_words = max([max(s) for s in x])

    if oov_char is not None:
        new_x = []
        for s in x:
            new_s = []
            for word in s:
                if word >= nb_words:
                    new_s.append(oov_char)
                else:
                    new_s.append(word)
            new_x.append(new_s)
    else:
        new_x = []
        for s in x:
            new_s = []
            for word in s:
                if word < nb_words:
                    new_s.append(word)
            new_x.append(new_s)
    x = new_x

    return (np.array(x[:len(x_train)]), np.array(y_train)), \
           (np.array(x[len(x_train):]), np.array(y_test))


def shuffle_by_seed(arr_list, seed=0):
    for arr in arr_list:
        np.random.seed(seed)
        np.random.shuffle(arr)


def get_word_index(dest_dir='/tmp/.bigdl/dataset', filename='imdb_word_index.pkl'):
    """Retrieves the dictionary mapping word indices back to words.

    # Arguments
        dest_dir: where to cache the data (relative to `~/.bigdl/dataset`).
        filename: dataset file name

    # Returns
        The word index dictionary.
    """

    path = base.maybe_download(filename,
                               dest_dir,
                               'https://s3.amazonaws.com/text-datasets/imdb_word_index.pkl',
                               )
    with open(path, 'rb') as f:
        data = cPickle.load(f, encoding='latin1')

    return data


if __name__ == "__main__":
    print('Processing text dataset')
    (x_train, y_train), (x_test, y_test) = load_data()
    print('finished processing text')
