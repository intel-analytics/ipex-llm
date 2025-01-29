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

import time
import os
import pandas as pd
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.friesian.feature import FeatureTable
from bigdl.dllib.feature.dataset import movielens
from bigdl.orca.learn.tf2.estimator import Estimator
from friesian.example.wnd.train.wnd_train_recsys import ColumnFeatureInfo, build_model
from bigdl.orca.data.file import exists, makedirs
import tensorflow as tf

data_dir = "./movielens"
model_dir = "./wnd"
wide_cols = ["gender", "age", "occupation", "zip"]
wide_cross_cols = ["gender_age", "age_zip"]
indicator_cols = ["gender", "age", "occupation"]
embed_cols = ["user", "item"]
num_cols = ["user_visits", "user_mean_rate", "item_visits", "item_mean_rate"]
cat_cols = wide_cols + wide_cross_cols + embed_cols
batch_size = 1024

if __name__ == '__main__':
    start = time.time()
    sc = init_orca_context("local",  cores=8, memory="8g", init_ray_on_spark=True)

    ratings = movielens.get_id_ratings(data_dir)
    ratings = pd.DataFrame(ratings, columns=["user", "item", "rate"])
    ratings_tbl = FeatureTable.from_pandas(ratings) \
        .cast(["user", "item", "rate"], "int")
    ratings_tbl.cache()

    user_tbl = FeatureTable.read_csv(data_dir + "/ml-1m/users.dat", delimiter=":")\
        .select("_c0", "_c2", "_c4", "_c6", "_c8")\
        .rename({"_c0": "user", "_c2": "gender", "_c4": "age", "_c6": "occupation", "_c8": "zip"})\
        .cast(["user"], "int")
    user_tbl.cache()

    user_tbl = user_tbl.fillna("0", "zip")

    user_stats = ratings_tbl.group_by("user", agg={"item": "count", "rate": "mean"}) \
        .rename({"count(item)": "user_visits", "avg(rate)": "user_mean_rate"})
    user_stats, user_min_max = user_stats.min_max_scale(["user_visits", "user_mean_rate"])

    item_stats = ratings_tbl.group_by("item", agg={"user": "count", "rate": "mean"}) \
        .rename({"count(user)": "item_visits", "avg(rate)": "item_mean_rate"})
    item_stats, item_min_max = item_stats.min_max_scale(["item_visits", "item_mean_rate"])

    user_tbl, inx_list = user_tbl.category_encode(["gender", "age", "zip", "occupation"])

    item_size = item_stats.select("item").distinct().size()
    ratings_tbl = ratings_tbl.add_negative_samples(item_size=item_size, item_col="item",
                                                   label_col="label", neg_num=1)

    user_tbl = user_tbl.cross_columns([["gender", "age"], ["age", "zip"]], [50, 200])

    user_tbl = user_tbl.join(user_stats, on="user")
    full = ratings_tbl.join(user_tbl, on="user").join(item_stats, on="item")
    stats = full.get_stats(cat_cols, "max")
    wide_dims = [stats[key] for key in wide_cols]
    wide_cross_dims = [stats[key] for key in wide_cross_cols]
    embed_dims = [stats[key] for key in embed_cols]
    indicator_dims = [stats[key] for key in indicator_cols]
    column_info = ColumnFeatureInfo(wide_base_cols=wide_cols,
                                    wide_base_dims=wide_dims,
                                    wide_cross_cols=wide_cross_cols,
                                    wide_cross_dims=wide_cross_dims,
                                    indicator_cols=indicator_cols,
                                    indicator_dims=indicator_dims,
                                    embed_cols=embed_cols,
                                    embed_in_dims=embed_dims,
                                    embed_out_dims=[8] * len(embed_dims),
                                    continuous_cols=num_cols,
                                    label="label")

    train_tbl, test_tbl = full.select("label", *column_info.feature_cols).random_split([0.8, 0.2])
    train_count, test_count = train_tbl.size(), test_tbl.size()

    def model_creator(conf):
        model = build_model(column_info=column_info,
                            hidden_units=[20, 10])
        optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['binary_accuracy', 'binary_crossentropy', 'AUC', 'Precision',
                               'Recall'])
        return model
    est = Estimator.from_keras(model_creator=model_creator)

    est.fit(data=train_tbl.df,
            epochs=1,
            batch_size=batch_size,
            steps_per_epoch=train_count // batch_size,
            validation_data=test_tbl.df,
            validation_steps=test_count // batch_size,
            feature_cols=column_info.feature_cols,
            label_cols=["label"])

    end = time.time()
    print("Training time is: ", end - start)
    model = est.get_model()
    if not exists(model_dir):
        makedirs(model_dir)
    model.save_weights(os.path.join(model_dir, "model.h5"))

    stop_orca_context()
