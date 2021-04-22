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

import sys
import os
import time
from pyspark import StorageLevel

try:
    import cPickle as pkl
except ModuleNotFoundError:
    import pickle as pkl

from optparse import OptionParser
from zoo.orca import init_orca_context, stop_orca_context, OrcaContext
from pyspark.sql.functions import udf, col
from zoo.friesian.feature import FeatureTable, StringIndex
from pyspark.sql.types import StringType, IntegerType, ArrayType, FloatType

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--meta", dest="meta_file")
    parser.add_option("--review", dest="review_file")
    parser.add_option("--output", dest="output")
    (options, args) = parser.parse_args(sys.argv)
    begin = time.time()
    sc = init_orca_context("local")
    spark = OrcaContext.get_spark_session()

    # read review datavi run.sh
    transaction_df = spark.read.json(options.review_file).select(
        ['reviewerID', 'asin', 'unixReviewTime']) \
        .withColumnRenamed('reviewerID', 'user') \
        .withColumnRenamed('asin', 'item') \
        .withColumnRenamed('unixReviewTime', 'time')\
        .dropna("any").persist(storageLevel=StorageLevel.DISK_ONLY)
    transaction_tbl = FeatureTable(transaction_df)
    print("review_tbl, ", transaction_tbl.count())

    # read meta data
    def get_category(x):
        cat = x[0][-1] if x[0][-1] is not None else "default"
        return cat.strip().lower()
    spark.udf.register("get_category", get_category, StringType())
    item_df = spark.read.json(options.meta_file).select(['asin', 'categories'])\
        .dropna(subset=['asin', 'categories']) \
        .selectExpr("*", "get_category(categories) as category") \
        .withColumnRenamed("asin", "item").drop("categories").distinct()\
        .persist(storageLevel=StorageLevel.DISK_ONLY)
    item_tbl = FeatureTable(item_df)

    print("item_tbl, ", item_tbl.count())

    item_category_indices = item_tbl.gen_string_idx(["item", "category"], 1)
    cat_default = item_category_indices[1].df.filter("category == 'default'").collect()
    default_cat = cat_default[0][1] if cat_default else item_category_indices[1].count()
    new_row = spark.createDataFrame([("default", int(default_cat))], ["category", "id"])
    category_index = StringIndex(item_category_indices[1].df.union(new_row).distinct()
                                 .withColumn("id", col("id").cast("Integer")), "category")
    item_size = item_category_indices[0].count()

    user_index = transaction_tbl.gen_string_idx(['user'], 1)
    get_label = udf(lambda x: [float(x), 1 - float(x)], ArrayType(FloatType()))
    item2cat = item_tbl\
        .encode_string(["item", "category"], [item_category_indices[0], category_index])\
        .distinct()

    full_tbl = transaction_tbl\
        .encode_string(['user', 'item'], [user_index[0], item_category_indices[0]])\
        .add_hist_seq(user_col="user", cols=['item'],
                      sort_col='time', min_len=1, max_len=100)\
        .add_neg_hist_seq(item_size, 'item_hist_seq', neg_num=5) \
        .add_negative_samples(item_size, item_col='item', neg_num=1)\
        .join(item2cat, "item")\
        .add_feature(["item_hist_seq", "neg_item_hist_seq"], item_tbl, default_cat)\
        .mask_pad(mask_cols=['item_hist_seq'],
                  padding_cols=['item_hist_seq', 'category_hist_seq',
                                'neg_item_hist_seq', 'neg_category_hist_seq'],
                  seq_len=100) \
        .add_length("item_hist_seq") \
        .transform_python_udf("label", "label", get_label)

    # write out
    user_index[0].write_parquet(options.output+"user_index")
    item_category_indices[0].write_parquet(options.output+"item_index")
    category_index.write_parquet(options.output+"category_index")
    item2cat.write_parquet(options.output+"item2cat")
    full_tbl.write_parquet(options.output + "data")

    print("final output count, ", full_tbl.count())
    stop_orca_context()
    end = time.time()
    print(f"perf preprocessing time: {(end - begin):.2f}s")
