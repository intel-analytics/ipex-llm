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
import random
import os
from argparse import ArgumentParser
from pyspark.sql.types import StructType, StructField, StringType, LongType, BooleanType
from bigdl.orca import init_orca_context, stop_orca_context, OrcaContext
from bigdl.friesian.feature import FeatureTable

id_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8",
           "9", "A", "B", "C", "D", "E", "F", "G", "H",
           "I", "J", "K", "L", "M", "N", "O", "P", "Q",
           "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
media_list = ["Photo", "Video", "GIF"]
tweet_list = ["Retweet", "Quote", "TopLevel"]
language_list = ["".join(random.choices(id_list, k=32)) for _ in range(65)]

schema = StructType(
    [StructField("text_tokens", StringType(), True),
     StructField("hashtags", StringType(), True),
     StructField("tweet_id", StringType(), True),
     StructField("present_media", StringType(), True),
     StructField("present_links", StringType(), True),
     StructField("present_domains", StringType(), True),
     StructField("tweet_type", StringType(), True),
     StructField("language", StringType(), True),
     StructField("tweet_timestamp", LongType(), True),
     StructField("engaged_with_user_id", StringType(), True),
     StructField("engaged_with_user_follower_count", LongType(), True),
     StructField("engaged_with_user_following_count", LongType(), True),
     StructField("engaged_with_user_is_verified", BooleanType(), True),
     StructField("engaged_with_user_account_creation", LongType(), True),
     StructField("enaging_user_id", StringType(), True),
     StructField("enaging_user_follower_count", LongType(), True),
     StructField("enaging_user_following_count", LongType(), True),
     StructField("enaging_user_is_verified", BooleanType(), True),
     StructField("enaging_user_account_creation", LongType(), True),
     StructField("engagee_follows_engager", StringType(), True),
     StructField("reply_timestamp", LongType(), True),
     StructField("retweet_timestamp", LongType(), True),
     StructField("retweet_with_comment_timestamp", LongType(), True),
     StructField("like_timestamp", LongType(), True)])


def _parse_args():
    parser = ArgumentParser(description="Generate random dataset for demo")
    parser.add_argument('num_samples', type=int,
                        help='The number of samples')
    parser.add_argument('output_path', type=str,
                        help='The path for output dataset')
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The cluster mode, such as local, yarn, standalone or spark-submit.')
    parser.add_argument('--master', type=str, default=None,
                        help='The master url, only used when cluster mode is standalone.')
    parser.add_argument('--executor_cores', type=int, default=48,
                        help='The executor core number.')
    parser.add_argument('--executor_memory', type=str, default="160g",
                        help='The executor memory.')
    parser.add_argument('--num_executors', type=int, default=8,
                        help='The number of executors.')
    parser.add_argument('--driver_cores', type=int, default=4,
                        help='The driver core number.')
    parser.add_argument('--driver_memory', type=str, default="36g",
                        help='The driver memory.')
    args = parser.parse_args()
    return args


def generate_record(random_seed):
    random.seed(random_seed)
    text_tokens = "\t".join([str(random.randint(1, 1000))
                            for i in range(random.randint(1, 10))])
    hashtags = "\t".join(["".join(random.choices(id_list, k=32))
                          for i in range(random.randint(0, 50))])
    tweet_id = "".join(random.choices(id_list, k=32))
    present_media = "\t".join(random.choices(
        media_list, k=random.randint(0, 9)))
    present_links = "\t".join(["".join(random.choices(id_list, k=32))
                               for i in range(random.randint(0, 10))])
    present_domains = "\t".join(["".join(random.choices(id_list, k=32))
                                for i in range(random.randint(0, 10))])
    tweet_type = random.choices(tweet_list)[0]
    language = random.choices(language_list)[0]
    tweet_timestamp = random.randint(946656000, 1609430400)
    engaged_with_user_id = "".join(random.choices(id_list, k=32))
    engaged_with_user_follower_count = random.randint(0, 10000)
    engaged_with_user_following_count = random.randint(0, 10000)
    engaged_with_user_is_verified = bool(random.getrandbits(1))
    engaged_with_user_account_creation = random.randint(946656000, 1609430400)
    enaging_user_id = "".join(random.choices(id_list, k=32))
    enaging_user_follower_count = random.randint(0, 10000)
    enaging_user_following_count = random.randint(0, 10000)
    enaging_user_is_verified = bool(random.getrandbits(1))
    enaging_user_account_creation = random.randint(946656000, 1609430400)
    engagee_follows_engager = bool(random.getrandbits(1))
    reply = bool(random.getrandbits(1))
    reply_timestamp = random.randint(946656000, 1609430400) if reply else None
    retweet = bool(random.getrandbits(1))
    retweet_timestamp = random.randint(
        946656000, 1609430400) if retweet else None
    comment = bool(random.getrandbits(1))
    retweet_with_comment_timestamp = random.randint(
        946656000, 1609430400) if comment else None
    like = bool(random.getrandbits(1))
    like_timestamp = random.randint(946656000, 1609430400) if like else None
    return (text_tokens, hashtags, tweet_id, present_media, present_links, present_domains,
            tweet_type, language, tweet_timestamp, engaged_with_user_id,
            engaged_with_user_follower_count, engaged_with_user_following_count,
            engaged_with_user_is_verified, engaged_with_user_account_creation,
            enaging_user_id, enaging_user_follower_count, enaging_user_following_count,
            enaging_user_is_verified, enaging_user_account_creation,
            engagee_follows_engager, reply_timestamp, retweet_timestamp,
            retweet_with_comment_timestamp, like_timestamp)


if __name__ == '__main__':
    args = _parse_args()
    if args.cluster_mode == "local":
        sc = init_orca_context("local", cores=args.executor_cores,
                               memory=args.executor_memory)
    elif args.cluster_mode == "standalone":
        sc = init_orca_context("standalone", master=args.master,
                               cores=args.executor_cores, num_nodes=args.num_executors,
                               memory=args.executor_memory,
                               driver_cores=args.driver_cores,
                               driver_memory=args.driver_memory)
    elif args.cluster_mode == "yarn":
        sc = init_orca_context("yarn-client", cores=args.executor_cores,
                               num_nodes=args.num_executors, memory=args.executor_memory,
                               driver_cores=args.driver_cores, driver_memory=args.driver_memory)
    elif args.cluster_mode == "spark-submit":
        sc = init_orca_context("spark-submit")
    else:
        argparse.ArgumentError(False,
                               "cluster_mode should be one of 'local', 'yarn', 'standalone' and"
                               " 'spark-submit', but got " + args.cluster_mode)
    spark = OrcaContext.get_spark_session()

    rdd = sc.parallelize(range(args.num_samples))
    dummy_data_rdd = rdd.map(generate_record)
    df = FeatureTable(spark.createDataFrame(dummy_data_rdd, schema))
    print(df.show(2))

    train_df, test_df = df.random_split([0.8, 0.2])
    print('train set size = ', train_df.size())
    print('test set size = ', test_df.size())

    train_df.write_parquet(os.path.join(args.output_path, 'train'))
    test_df.write_parquet(os.path.join(args.output_path, 'test'))
    print('Save data finished')

    stop_orca_context()
