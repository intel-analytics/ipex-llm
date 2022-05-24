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

import os
import sys
import tensorflow as tf
import tensorflow_ranking as tfr
from optparse import OptionParser

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca import OrcaContext
from bigdl.friesian.feature import FeatureTable
from bigdl.orca.learn.tf2 import Estimator

from models import Padded2RaggedModel

# OrcaContext._shard_size = 10000

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('--cluster_mode', type=str, default="local",
                      help='The cluster mode, such as local, yarn or standalone.')
    parser.add_option('--executor_cores', type=int, default=18,
                      help='The executor core number.')
    parser.add_option('--executor_memory', type=str, default="10g",
                      help='The executor memory.')
    parser.add_option('--num_executor', type=int, default=1,
                      help='The number of executor.')
    parser.add_option('--driver_cores', type=int, default=4,
                      help='The driver core number.')
    parser.add_option('--driver_memory', type=str, default="36g",
                      help='The driver memory.')
    parser.add_option("--data_dir", dest="data_dir", default="./total.parquet",
                      help='The directory to read/write data')

    (options, args) = parser.parse_args(sys.argv)
    data_dir = options.data_dir

    if options.cluster_mode == "local":
        init_orca_context("local", cores=options.executor_cores, memory=options.executor_memory,
                          init_ray_on_spark=True)
    elif options.cluster_mode == "yarn":
        init_orca_context("yarn-client", cores=options.executor_cores,
                          num_nodes=options.num_executor, memory=options.executor_memory,
                          driver_cores=options.driver_cores, driver_memory=options.driver_memory,
                          init_ray_on_spark=True)
    else:
        raise ValueError("cluster_mode should be 'local' or 'yarn', but got " + args.cluster_mode)

    (options, args) = parser.parse_args(sys.argv)
    data_dir = options.data_dir

    dataset = {
        "ratings": ['userid', 'movieid', 'rating', 'timestamp'],
        "users": ["userid", "gender", "age", "occupation", "zip-code"],
        "movies": ["movieid", "title", "genres"]
    }

    tbl_dict = dict()
    for data, cols in dataset.items():
        tbl = FeatureTable.read_csv(os.path.join(data_dir, data + ".dat"),
                                    delimiter=":", header=False)
        tmp_cols = tbl.columns[::2]
        tbl = tbl.select(tmp_cols)
        col_dict = {c[0]: c[1] for c in zip(tmp_cols, cols)}
        tbl = tbl.rename(col_dict)
        tbl_dict[data] = tbl

    full_tbl = tbl_dict["ratings"].join(tbl_dict["movies"], "movieid") \
        .dropna(columns=None).select(["userid", "title", "rating"])
    # cast
    full_tbl = full_tbl.cast(["rating"], "int")
    full_tbl = full_tbl.cast(["userid"], "string")
    train_tbl, test_tbl = full_tbl.random_split([0.85, 0.15], seed=1)


    def preprocess(tbl, feature_cols):
        col_dict = {"collect_list(" + c + ")": c + "s" for c in feature_cols}
        tbl = tbl.group_by("userid", agg="collect_list")
        tbl = tbl.rename(col_dict)
        arr_len = lambda x: len(x)
        tbl = tbl.apply("ratings", "len", arr_len, dtype="int")
        min_len = tbl.get_stats("len", "min")["len"]
        max_len = tbl.get_stats("len", "max")["len"]
        print("max_min_len", (max_len, min_len))
        tbl = tbl.pad(["ratings"], max_len, mask_token=-1)
        tbl = tbl.pad(["titles"], max_len, mask_token="<MSK>")
        return tbl, max_len


    feature_cols = ["title", "rating"]
    train_tbl, max_len = preprocess(train_tbl, feature_cols)
    test_tbl, _ = preprocess(test_tbl, feature_cols)
    print(train_tbl.schema)
    train_tbl.sort("len", ascending=False).show(3, False)

    unique_movie_titles = tbl_dict["movies"].get_vocabularies(["title"])["title"]
    tbl_dict["users"] = tbl_dict["users"].cast("userid", "string")
    unique_userids = tbl_dict["users"].get_vocabularies(["userid"])["userid"]
    print(len(unique_movie_titles), len(unique_userids))
    print(unique_movie_titles[0:2])
    print(unique_userids[0:2])

    model_config = {
        "learning_rate": 0.1,
        "userid_vocab": unique_userids,
        "movie_vocab": unique_movie_titles,
        "max_len": max_len,
    }


    def create_model(config):
        # loss = tfr.keras.losses.MeanSquaredLoss(ragged=True)
        loss = tfr.keras.losses.ListMLELoss(ragged=True)
        # loss = tfr.keras.losses.PairwiseHingeLoss(ragged=True)
        # loss = tfr.keras.losses.ApproxNDCGLoss(ragged=True)

        model = Padded2RaggedModel(
            config["userid_vocab"],
            config["movie_vocab"],
            config["max_len"],
            loss
        )
        from bigdl.friesian.models import TFRSModel
        model = TFRSModel(model)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
        )

        return model


    est = Estimator.from_keras(
        model_creator=create_model,
        config=model_config,
        workers_per_node=1
    )

    train_count = train_tbl.size()
    test_count = test_tbl.size()

    batch_size = 256
    train_steps = train_count // batch_size
    test_steps = test_count // batch_size

    print(train_count, train_steps)
    print(test_count, test_steps)

    est.fit(train_tbl.df, epochs=16,
            batch_size=batch_size,
            feature_cols=["userid", "titles", "len"],
            label_cols=["ratings"],
            steps_per_epoch=train_steps,
            validation_data=test_tbl.df,
            validation_steps=test_steps)

    stop_orca_context()
