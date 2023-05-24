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

import argparse
from model import *
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.tf2 import Estimator
from bigdl.friesian.feature import FeatureTable

parser = argparse.ArgumentParser(description="Two Tower Inference")
parser.add_argument('--cluster_mode', type=str, default="local",
                    help='The cluster mode, such as local, yarn or spark-submit.')
parser.add_argument('--backend', type=str, default="ray",
                    choices=("spark", "ray"),
                    help="The backend of Orca Estimator, either ray or spark.")
parser.add_argument('--num_executors', type=int, default=8,
                    help='The number of executors.')
parser.add_argument('--executor_cores', type=int, default=8,
                    help="The executor core number.")
parser.add_argument('--executor_memory', type=str, default="2g",
                    help="The executor memory.")
parser.add_argument('--batch_size', default=8000, type=int, help='batch size')
parser.add_argument('--model_dir', default='recsys_2tower', type=str,
                    help="model directory name (default: recsys_2tower)")
parser.add_argument('--data_dir', type=str,
                    help="data directory of processed features for the two tower model")
args = parser.parse_args()

if args.cluster_mode == "local":
    sc = init_orca_context("local", cores=args.executor_cores,
                           memory=args.executor_memory)
elif args.cluster_mode == "yarn":
    sc = init_orca_context("yarn-client", cores=args.executor_cores,
                           num_nodes=args.num_executors, memory=args.executor_memory,
                           extra_python_lib="model.py")
elif args.cluster_mode == "spark-submit":
    sc = init_orca_context("spark-submit")
else:
    invalidInputError(False,
                      "cluster_mode should be one of 'local', 'yarn' and"
                      " 'spark-submit', but got " + args.cluster_mode)

config = {"inter_op_parallelism": 4,
          "intra_op_parallelism": args.executor_cores}

user_est = Estimator.from_keras(config=config, backend=args.backend)
user_est.load(os.path.join(args.model_dir, "user-model"))

full_tbl = FeatureTable.read_parquet(os.path.join(args.data_dir, "user_item_parquet"))
print("Data size: " + str(full_tbl.size()))

enaging_user_df = full_tbl.select(['enaging_user_id', 'enaging_user_is_verified',
                                   'user_numeric'])\
                          .rename({'enaging_user_id': 'user_id',
                                   'enaging_user_is_verified': 'is_verified'})
# the last 3 columns of "item_numeric" are engaged users' numeric features
engaged_user_df = full_tbl.select(['engaged_with_user_id', 'engaged_with_user_is_verified',
                                   'item_numeric'])\
                          .apply("item_numeric", "user_numeric",
                                 lambda item_numeric: item_numeric[-3:],
                                 dtype="array<float>")\
                          .rename({'engaged_with_user_id': 'user_id',
                                   'engaged_with_user_is_verified': 'is_verified'})
user_df = enaging_user_df.concat(engaged_user_df)

user_embed = FeatureTable(user_est.predict(data=user_df.df,
                                           batch_size=args.batch_size,
                                           feature_cols=user_df.columns))\
    .select(['user_id', 'prediction']).drop_duplicates()
print("Embeddings of the first 5 users:")
user_embed.show(5)
user_embed.write_parquet(os.path.join(args.data_dir, 'user_ebd.parquet'))

item_est = Estimator.from_keras(config=config, backend=args.backend)
item_est.load(os.path.join(args.model_dir, "item-model"))
item_embed = item_est.predict(data=full_tbl.df,
                              batch_size=args.batch_size,
                              feature_cols=['engaged_with_user_is_verified', 'present_media',
                                            'tweet_type', 'language', 'tweet_id',
                                            'engaged_with_user_id', 'hashtags',
                                            'present_links', 'present_domains', 'item_numeric'])
item_embed = FeatureTable(item_embed)
item_embed = item_embed.select(['tweet_id', 'prediction']).drop_duplicates()
print("Embeddings of the first 5 items:")
item_embed.show(5)
item_embed.write_parquet(os.path.join(args.data_dir, 'item_ebd.parquet'))

stop_orca_context()
