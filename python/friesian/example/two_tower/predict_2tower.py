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

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.tf2 import Estimator
from bigdl.friesian.feature import FeatureTable
from model import *
import argparse

parser = argparse.ArgumentParser(description='Two Tower Training/Inference')
parser.add_argument('--backend', type=str, default="ray",
                    choices=("spark", "ray"),
                    help='The backend of Orca Estimator, either ray or spark.')
parser.add_argument('--model_dir', default='recsys_2tower', type=str,
                    help='model directory name (default: recsys_2tower)')
parser.add_argument('--executor_cores', type=int, default=8,
                    help='The executor core number.')
parser.add_argument('--executor_memory', type=str, default="2g",
                        help='The executor memory.')
parser.add_argument('--data_dir', type=str, help='data directory')
args = parser.parse_args()

sc = init_orca_context("local", cores=args.executor_cores, memory=args.executor_memory)

train_config = {"inter_op_parallelism": 4,
                "intra_op_parallelism": args.executor_cores}

user_est = Estimator.from_keras(config=train_config,
                                backend=args.backend)
user_est.load(os.path.join(args.model_dir, "user-model"))

full_tbl = FeatureTable.read_parquet(os.path.join(args.model_dir, "full_parquet"))
print("full table size: "+str(full_tbl.size()))

result = user_est.predict(data=full_tbl.df,
                          feature_cols=['enaging_user_is_verified', 'enaging_user_id', 'user_num'])
print("Prediction results of the first 5 rows:")
result.show(5)

result = FeatureTable(result)
result = result.select(['enaging_user_id', 'prediction']).drop_duplicates()
result.write_parquet(os.path.join(args.model_dir, 'user_ebd.parquet'))

del result, user_est

item_est = Estimator.from_keras(config=train_config,
                                backend=args.backend)
item_est.load(os.path.join(args.model_dir, "item-model"))
result = item_est.predict(data=full_tbl.df,
                          feature_cols=['engaged_with_user_is_verified', 'present_media', 'tweet_type',
                                        'language', 'tweet_id', 'engaged_with_user_id', 'hashtags',
                                        'present_links', 'present_domains', 'item_num'])

print("Prediction results of the first 5 rows:")
result.show(5)
result = FeatureTable(result)
result = result.select(['tweet_id', 'prediction']).drop_duplicates()
result.write_parquet(os.path.join(args.model_dir, 'item_ebd.parquet'))

stop_orca_context()
