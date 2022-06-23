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
import pandas as pd
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.friesian.feature import FeatureTable
from bigdl.dllib.feature.dataset import movielens
from bigdl.orca.learn.tf.estimator import Estimator
from friesian.example.dien.dien_train import build_model
import time
from bigdl.orca.data.file import exists, makedirs
from pyspark.sql.functions import lit


data_dir = "./movielens"
model_dir = "./dien"
batch_size = 32
seq_length = 50

if __name__ == '__main__':
    start = time.time()
    sc = init_orca_context("local",  cores=8, memory="8g", init_ray_on_spark=True)

    _ = movielens.get_id_ratings(data_dir)
    ratings = pd.read_csv(data_dir + "/ml-1m/ratings.dat", delimiter="::",
                          names=["user", "item", "rate", "time"])
    ratings = pd.DataFrame(ratings, columns=["user", "item", "rate", "time"])
    ratings_tbl = FeatureTable.from_pandas(ratings) \
        .cast(["user", "item", "rate"], "int").cast("time", "long")
    ratings_tbl.cache()

    item_df = pd.read_csv(data_dir + "/ml-1m/movies.dat", encoding="ISO-8859-1",
                          delimiter="::", names=["item", "title", "genres"])
    item_tbl = FeatureTable.from_pandas(item_df).drop("title").\
        rename({"genres": "category"}).cast("item", "int")

    item_tbl.cache()

    cat_indx = item_tbl.gen_string_idx("category", freq_limit=1)
    item_tbl = item_tbl.encode_string("category", cat_indx)
    user_size = ratings_tbl.get_stats("user", "max")["user"] + 1
    item_size = ratings_tbl.get_stats("item", "max")["item"] + 1
    cat_size = item_tbl.get_stats("category", "max")["category"] + 1

    full_tbl = ratings_tbl \
        .add_hist_seq(cols=['item'], user_col="user",
                      sort_col='time', min_len=1, max_len=seq_length, num_seqs=1) \
        .add_neg_hist_seq(item_size, 'item_hist_seq', neg_num=5) \
        .add_negative_samples(item_size, item_col='item', neg_num=1) \
        .add_value_features(columns=["item", "item_hist_seq", "neg_item_hist_seq"],
                            dict_tbl=item_tbl, key="item", value="category") \
        .pad(cols=['item_hist_seq', 'category_hist_seq',
                   'neg_item_hist_seq', 'neg_category_hist_seq'],
             seq_len=seq_length,
             mask_cols=['item_hist_seq']) \
        .append_column("item_hist_seq_len", lit(seq_length)) \
        .apply("label", "label", lambda x: [1 - float(x), float(x)], "array<float>")

    train_tbl, test_tbl = full_tbl.split([0.8, 0.2], seed=1)

    model = build_model("DIEN", user_size, item_size, cat_size, 0.001, "FP32")

    input_phs = [model.uid_batch_ph, model.mid_his_batch_ph, model.cat_his_batch_ph, model.mask,
                 model.seq_len_ph, model.mid_batch_ph, model.cat_batch_ph,
                 model.noclk_mid_batch_ph, model.noclk_cat_batch_ph]
    feature_cols = ['user', 'item_hist_seq', 'category_hist_seq', 'item_hist_seq_mask',
                    'item_hist_seq_len', 'item', 'category',
                    'neg_item_hist_seq', 'neg_category_hist_seq']

    estimator = Estimator.from_graph(inputs=input_phs, outputs=[model.y_hat],
                                     labels=[model.target_ph], loss=model.loss,
                                     optimizer=model.optim, model_dir=model_dir,
                                     metrics={'loss': model.loss, 'accuracy': model.accuracy})

    estimator.fit(train_tbl.df, epochs=1, batch_size=batch_size,
                  feature_cols=feature_cols, label_cols=['label'], validation_data=test_tbl.df)

    if not exists(model_dir):
        makedirs(model_dir)
    estimator.save_tf_checkpoint(model_dir)

    end = time.time()
    print(f"DIEN processing time: {(end - start):.2f}s")

    stop_orca_context()
