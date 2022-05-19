import os
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_ranking as tfr
import argparse

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca import OrcaContext
from pyspark.sql.types import ArrayType, IntegerType, StringType
from bigdl.friesian.feature import FeatureTable
from bigdl.orca.learn.tf2 import Estimator

from models import Padded2RaggedModel

# OrcaContext._shard_size = 10000

cluster_mode = "local"
# data_path = "/home/yina/Documents/data/movielen/ml-1m"
data_path = "/Users/yita/Documents/intel/data"

if cluster_mode == "local":
    init_orca_context(cluster_mode="local", cores=8, memory="6g")
elif cluster_mode == "yarn":
    init_orca_context(cluster_mode="yarn-client", num_nodes=2, cores=2)

dataset = {
    "ratings": ['userid', 'movieid',  'rating', 'timestamp'],
    "users": ["userid", "gender", "age", "occupation", "zip-code"],
    "movies": ["movieid", "title", "genres"]
}

tbl_dict = dict()
for data, cols in dataset.items():
    tbl = FeatureTable.read_csv(os.path.join(data_path, data + ".dat"),
                                delimiter=":", header=False)
    tmp_cols = tbl.columns[::2]
    tbl = tbl.select(tmp_cols)
    col_dict = {c[0]: c[1] for c in zip(tmp_cols, cols)}
    tbl = tbl.rename(col_dict)
    tbl_dict[data] = tbl

full_tbl = tbl_dict["ratings"].join(tbl_dict["movies"], "movieid")\
    .dropna(columns=None).select(["userid", "title", "rating"])
full_tbl = full_tbl.limit(1000).repartition(4)
# cast
full_tbl = full_tbl.cast(["rating"], "int")
full_tbl = full_tbl.cast("userid", "string")
train_tbl, test_tbl = full_tbl.random_split([0.85, 0.15], seed=1)

train_tbl = train_tbl.group_by("userid", agg="collect_list")
test_tbl = test_tbl.group_by("userid", agg="collect_list")

feature_cols = ["title", "rating"]
col_dict = {"collect_list(" + c + ")": c +"s" for c in feature_cols}
train_tbl = train_tbl.rename(col_dict)
test_tbl = test_tbl.rename(col_dict)
print(train_tbl.schema)
train_tbl.show(2)

unique_movie_titles = tbl_dict["movies"].get_vocabularies(["title"])["title"]
tbl_dict["users"] = tbl_dict["users"].cast("userid", "string")
unique_userids = tbl_dict["users"].get_vocabularies(["userid"])["userid"]
print(len(unique_movie_titles), len(unique_userids))
print(unique_movie_titles[0:2])
print(unique_userids[0:2])

arr_count = lambda x: len(x)
train_tbl = train_tbl.apply("ratings", "len", arr_count, dtype="int")
test_tbl = test_tbl.apply("ratings", "len", arr_count, dtype="int")

min_len = train_tbl.get_stats("len", "min")["len"]
max_len = train_tbl.get_stats("len", "max")["len"]
print("max_min_len", (max_len, min_len))


def pad_list(lst, seq_len, mask_token=1):
    size = len(lst)
    lst.extend([mask_token] * (seq_len - size))
    return lst


train_tbl = train_tbl.apply("ratings", "pad_ratings", lambda x: pad_list(x, max_len, -1),
                            ArrayType(IntegerType()))
train_tbl = train_tbl.apply("titles", "pad_titles", lambda x: pad_list(x, max_len, "<MSK>"),
                            ArrayType(StringType()))
train_tbl = train_tbl.drop("ratings", "titles")
test_tbl = test_tbl.apply("ratings", "pad_ratings", lambda x: pad_list(x, max_len, -1),
                          ArrayType(IntegerType()))
test_tbl = test_tbl.apply("titles", "pad_titles", lambda x: pad_list(x, max_len, "<MSK>"),
                          ArrayType(StringType()))
test_tbl = test_tbl.drop("ratings", "titles")

model_config = {
    "learning_rate": 0.1,
    "userid_vocab": unique_userids,
    "movie_vocab": unique_movie_titles,
    "max_len": max_len,
#     "test_data": test_data,
}

from bigdl.orca.learn.utils import _dataframe_to_xshards
from bigdl.orca.data.utils import ray_partition_get_data_label
train_tbl = train_tbl.repartition(6)
train_tbl.get_partition_row_number().show()
train_shard = _dataframe_to_xshards(train_tbl.df, ["userid", "pad_titles", "len"], ["pad_ratings"],
                                    accept_str_col=True)
a = train_shard.collect()
data, label = ray_partition_get_data_label(a)
train = tf.data.Dataset.from_tensor_slices((data, label))
train = train.batch(2)

hinge_model = Padded2RaggedModel(loss=tfr.keras.losses.PairwiseHingeLoss(ragged=True),
                                 max_len=max_len, unique_userids=unique_userids,
                                 unique_movie_titles=unique_movie_titles)
hinge_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
hinge_model.fit(train, epochs=1)

