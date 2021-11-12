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

from bigdl.dataset import movielens
from bigdl.orca.learn.tf2.estimator import Estimator
from bigdl.friesian.feature import FeatureTable
from bigdl.orca import init_orca_context, stop_orca_context
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, concatenate, multiply
import tensorflow as tf
import math
import pandas as pd
import argparse


def build_model(num_users, num_items, class_num, layers=[20, 10], include_mf=True, mf_embed=20):
    num_layer = len(layers)
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    mlp_embed_user = Embedding(input_dim=num_users, output_dim=int(layers[0] / 2),
                               input_length=1)(user_input)
    mlp_embed_item = Embedding(input_dim=num_items, output_dim=int(layers[0] / 2),
                               input_length=1)(item_input)

    user_latent = Flatten()(mlp_embed_user)
    item_latent = Flatten()(mlp_embed_item)

    mlp_lalent = concatenate([user_latent, item_latent], axis=1)
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], activation='relu',
                      name='layer%d' % idx)
        mlp_lalent = layer(mlp_lalent)

    if include_mf:
        mf_embed_user = Embedding(input_dim=num_users,
                                  output_dim=mf_embed,
                                  input_length=1)(user_input)
        mf_embed_item = Embedding(input_dim=num_users,
                                  output_dim=mf_embed,
                                  input_length=1)(item_input)
        mf_user_flatten = Flatten()(mf_embed_user)
        mf_item_flatten = Flatten()(mf_embed_item)

        mf_latent = multiply([mf_user_flatten, mf_item_flatten])
        concated_model = concatenate([mlp_lalent, mf_latent], axis=1)
        prediction = Dense(class_num, activation="softmax")(concated_model)
    else:
        prediction = Dense(class_num, activation='softmax', name='prediction')(mlp_lalent)

    model = tf.keras.Model([user_input, item_input], prediction)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Two Tower Training/Inference')
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The cluster mode, such as local, yarn or standalone.')
    parser.add_argument('--master', type=str, default=None,
                        help='The master url, only used when cluster mode is standalone.')
    parser.add_argument('--executor_cores', type=int, default=8,
                        help='The executor core number.')
    parser.add_argument('--executor_memory', type=str, default="160g",
                        help='The executor memory.')
    parser.add_argument('--num_executor', type=int, default=8,
                        help='The number of executor.')
    parser.add_argument('--driver_cores', type=int, default=4,
                        help='The driver core number.')
    parser.add_argument('--driver_memory', type=str, default="36g",
                        help='The driver memory.')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=5, type=int, help='train epoch')
    parser.add_argument('--batch_size', default=8000, type=int, help='batch size')
    parser.add_argument('--model_dir', default='snapshot', type=str,
                        help='snapshot directory name (default: snapshot)')
    args = parser.parse_args()

    if args.cluster_mode == "local":
        sc = init_orca_context("local", init_ray_on_spark=True)
    elif args.cluster_mode == "standalone":
        sc = init_orca_context("standalone", master=args.master,
                               cores=args.executor_cores, num_nodes=args.num_executor,
                               memory=args.executor_memory,
                               driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                               init_ray_on_spark=True)
    elif args.cluster_mode == "yarn":
        sc = init_orca_context("yarn-client", cores=args.executor_cores,
                               num_nodes=args.num_executor, memory=args.executor_memory,
                               driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                               object_store_memory="10g",
                               init_ray_on_spark=True)
    elif args.cluster_mode == "spark-submit":
        sc = init_orca_context("spark-submit")

    movielens_data = movielens.get_id_ratings("/tmp/movielens/")
    pddf = pd.DataFrame(movielens_data, columns=["user", "item", "label"])
    num_users, num_items = pddf["user"].max() + 1, pddf["item"].max() + 1

    full = FeatureTable.from_pandas(pddf)\
        .apply("label", "label", lambda x: x - 1, 'int')
    train, test = full.random_split([0.8, 0.2], seed=1)

    config = {"lr": 1e-3, "inter_op_parallelism": 4, "intra_op_parallelism": args.executor_cores}

    def model_creator(config):
        model = build_model(num_users, num_items, 5)
        print(model.summary())
        optimizer = tf.keras.optimizers.Adam(config["lr"])
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_crossentropy', 'accuracy'])
        return model

    steps_per_epoch = math.ceil(train.size() / args.batch_size)
    val_steps = math.ceil(test.size() / args.batch_size)

    estimator = Estimator.from_keras(model_creator=model_creator,
                                     verbose=False,
                                     config=config)
    estimator.fit(train.df,
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  feature_cols=['user', 'item'],
                  label_cols=['label'],
                  steps_per_epoch=steps_per_epoch,
                  validation_data=test.df,
                  validation_steps=val_steps)

    tf.saved_model.save(estimator.get_model(), args.model_dir)

    stop_orca_context()
