from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.tf2 import Estimator
from bigdl.friesian.feature import FeatureTable
from model import *
import argparse

sc = init_orca_context()

parser = argparse.ArgumentParser(description='Two Tower Training/Inference')
parser.add_argument('--backend', type=str, default="spark",
                    choices=("spark", "ray"),
                    help='The backend of Orca Estimator, either ray or spark.')
parser.add_argument('--model_dir', default='recsys_2tower', type=str,
                    help='snapshot directory name (default: snapshot)')
parser.add_argument('--executor_cores', type=int, default=8,
                    help='The executor core number.')
parser.add_argument('--data_dir', type=str, help='data directory')
parser.add_argument('--frequency_limit', type=int, default=25, help='frequency limit')
args = parser.parse_args()


train_config = {"lr": 1e-3,
                "inter_op_parallelism": 4,
                "intra_op_parallelism": args.executor_cores}

user_est = Estimator.from_keras(config=train_config,
                                backend=args.backend)
user_est.load(os.path.join(args.model_dir, "user-model"))

full_tbl = FeatureTable.read_parquet(os.path.join(args.model_dir, "full_parquet"))
print("full size: "+str(full_tbl.size()))

result = user_est.predict(data=full_tbl.df,
                          feature_cols=['enaging_user_is_verified', 'enaging_user_id', 'user_num'])
print("Prediction results of the first 5 rows:")
result.show(5)

result = FeatureTable(result)
result = result.select(['enaging_user_id', 'prediction']).drop_duplicates()
result.write_parquet(os.path.join(args.model_dir, 'user_ebd.parquet'))
print("user columns: "+ str(result.columns))
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
print("item columns: "+ str(result.columns))

stop_orca_context()
