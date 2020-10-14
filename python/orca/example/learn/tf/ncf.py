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
import os
import zipfile
import argparse

import numpy as np
import tensorflow as tf

from bigdl.dataset import base
from sklearn.model_selection import train_test_split

from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca.learn.tf.estimator import Estimator
from zoo.orca.data import SharedValue
import zoo.orca.data.pandas

SOURCE_URL = 'http://files.grouplens.org/datasets/movielens/'
COLUMN_NAMES = ['user', 'item', 'label']


def re_index(s):
    """ for reindexing the item set. """
    i = 0
    s_map = {}
    for key in s:
        s_map[key] = i
        i += 1

    return s_map


def set_index(data, user_map, item_map):
    def set_user_item(df, item_map, user_map):
        user_list = []
        item_list = []
        item_map = item_map.value
        user_map = user_map.value
        for i in range(len(df)):
            user_list.append(user_map[df['user'][i]])
            item_list.append(item_map[df['item'][i]])
        df['user'] = user_list
        df['item'] = item_list
        return df

    user_map_shared_value = SharedValue(user_map)
    item_map_shared_value = SharedValue(item_map)
    return data.transform_shard(set_user_item, item_map_shared_value, user_map_shared_value)


def load_data(data_dir):
    WHOLE_DATA = 'ml-1m.zip'
    local_file = base.maybe_download(WHOLE_DATA, data_dir, SOURCE_URL + WHOLE_DATA)
    zip_ref = zipfile.ZipFile(local_file, 'r')
    extracted_to = os.path.join(data_dir, "ml-1m")
    if not os.path.exists(extracted_to):
        print("Extracting %s to %s" % (local_file, data_dir))
        zip_ref.extractall(data_dir)
        zip_ref.close()
    rating_files = os.path.join(extracted_to, "ratings.dat")

    # replace :: to : for spark 2.4 support
    new_rating_files = os.path.join(extracted_to, "ratings_new.dat")
    if not os.path.exists(new_rating_files):
        fin = open(rating_files, "rt")
        # output file to write the result to
        fout = open(new_rating_files, "wt")
        # for each line in the input file
        for line in fin:
            # read replace the string and write to output file
            fout.write(line.replace('::', ':'))
        # close input and output files
        fin.close()
        fout.close()

    # read movive len csv to XShards of Pandas Dataframe
    full_data = zoo.orca.data.pandas.read_csv(new_rating_files, sep=':', header=None,
                                              names=COLUMN_NAMES, usecols=[0, 1, 2],
                                              dtype={0: np.int32, 1: np.int32, 2: np.int32})

    user_set = set(full_data['user'].unique())
    item_set = set(full_data['item'].unique())

    min_user_id = min(user_set)
    max_user_id = max(user_set)
    min_item_id = min(item_set)
    max_item_id = max(item_set)
    print(min_user_id, max_user_id, min_item_id, max_item_id)

    # update label starting from 0
    def update_label(df):
        df['label'] = df['label'] - 1
        return df

    full_data = full_data.transform_shard(update_label)

    # split to train/test dataset
    def split_train_test(data):
        # splitting the full set into train and test sets.
        train, test = train_test_split(data, test_size=0.2, random_state=100)
        return train, test

    train_data, test_data = full_data.transform_shard(split_train_test).split()

    def to_train_val_shard(df):
        result = {
            "x": (df['user'].to_numpy(), df['item'].to_numpy()),
            "y": df['label'].to_numpy()
        }
        return result

    train_data = train_data.transform_shard(to_train_val_shard)
    test_data = test_data.transform_shard(to_train_val_shard)
    return train_data, test_data, max_user_id, max_item_id


class NCF(object):
    def __init__(self, embed_size, user_size, item_size):
        self.user = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.item = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.label = tf.placeholder(dtype=tf.int32, shape=(None,))

        with tf.name_scope("GMF"):
            user_embed_GMF = tf.contrib.layers.embed_sequence(self.user,
                                                              vocab_size=user_size + 1,
                                                              embed_dim=embed_size,
                                                              unique=False
                                                              )
            item_embed_GMF = tf.contrib.layers.embed_sequence(self.item,
                                                              vocab_size=item_size + 1,
                                                              embed_dim=embed_size,
                                                              unique=False
                                                              )
            GMF = tf.multiply(user_embed_GMF, item_embed_GMF, name='GMF')

        # MLP part starts
        with tf.name_scope("MLP"):
            user_embed_MLP = tf.contrib.layers.embed_sequence(self.user,
                                                              vocab_size=user_size + 1,
                                                              embed_dim=embed_size,
                                                              unique=False,
                                                              )

            item_embed_MLP = tf.contrib.layers.embed_sequence(self.item,
                                                              vocab_size=item_size + 1,
                                                              embed_dim=embed_size,
                                                              unique=False
                                                              )
            interaction = tf.concat([user_embed_MLP, item_embed_MLP],
                                    axis=-1, name='interaction')

            layer1_MLP = tf.layers.dense(inputs=interaction,
                                         units=embed_size * 2,
                                         name='layer1_MLP')
            layer1_MLP = tf.layers.dropout(layer1_MLP, rate=0.2)

            layer2_MLP = tf.layers.dense(inputs=layer1_MLP,
                                         units=embed_size,
                                         name='layer2_MLP')
            layer2_MLP = tf.layers.dropout(layer2_MLP, rate=0.2)

            layer3_MLP = tf.layers.dense(inputs=layer2_MLP,
                                         units=embed_size // 2,
                                         name='layer3_MLP')
            layer3_MLP = tf.layers.dropout(layer3_MLP, rate=0.2)

        # Concate the two parts together
        with tf.name_scope("concatenation"):
            concatenation = tf.concat([GMF, layer3_MLP], axis=-1,
                                      name='concatenation')
            self.logits = tf.layers.dense(inputs=concatenation,
                                          units=5,
                                          name='predict')

            self.logits_softmax = tf.nn.softmax(self.logits)

            self.class_number = tf.argmax(self.logits_softmax, 1)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.label, logits=self.logits, name='loss'))

        with tf.name_scope("optimzation"):
            self.optim = tf.train.AdamOptimizer(1e-3, name='Adam')
            self.optimizer = self.optim.minimize(self.loss)


def train(train_data, test_data, user_size, item_size):
        model = NCF(opt.embedding_size, user_size, item_size)

        estimator = Estimator.from_graph(
            inputs=[model.user, model.item],
            outputs=[model.class_number],
            labels=[model.label],
            loss=model.loss,
            optimizer=model.optim,
            model_dir=opt.model_dir,
            metrics={"loss": model.loss})

        estimator.fit(data=train_data,
                      batch_size=opt.batch_size,
                      epochs=opt.epochs,
                      validation_data=test_data
                      )

        checkpoint_path = os.path.join(opt.model_dir, "NCF.ckpt")
        estimator.save_tf_checkpoint(checkpoint_path)
        estimator.sess.close()


def predict(predict_data, user_size, item_size):

    def to_predict(data):
        del data['y']
        return data

    predict_data = predict_data.transform_shard(to_predict)

    tf.reset_default_graph()

    with tf.Session() as sess:
        model = NCF(opt.embedding_size, user_size, item_size)

        saver = tf.train.Saver(tf.global_variables())
        checkpoint_path = os.path.join(opt.model_dir, "NCF.ckpt")
        saver.restore(sess, checkpoint_path)

        estimator = Estimator.from_graph(
            inputs=[model.user, model.item],
            outputs=[model.class_number],
            sess=sess,
            model_dir=opt.model_dir
        )
        predict_result = estimator.predict(predict_data)
        predictions = predict_result.collect()
        assert 'prediction' in predictions[0]
        print(predictions[0]['prediction'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NCF example on movie len dataset.')
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The mode for the Spark cluster. local or yarn.')
    parser.add_argument('--data_dir', type=str, default='/tmp',
                        help='the dir for downloaded data.')
    parser.add_argument('--embedding_size', type=int, default=16,
                        help='the size for embedding user and item.')
    parser.add_argument('--model_dir', type=str, default='./',
                        help='the dir for saving model.')
    parser.add_argument('-b', '--batch_size', type=int, default=1280,
                        help='size of mini-batch')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='The number of epochs to train the model.')

    opt = parser.parse_args()
    if opt.cluster_mode == "local":
        init_orca_context(cluster_mode="local", cores=4)
    elif opt.cluster_mode == "yarn":
        init_orca_context(cluster_mode="yarn-client", num_nodes=2, cores=2, driver_memory="6g")

    (train_data, test_data, max_user_id, max_item_id) = load_data(opt.data_dir)

    train(train_data, test_data, max_user_id, max_item_id)

    predict(test_data, max_user_id, max_item_id)

    stop_orca_context()
