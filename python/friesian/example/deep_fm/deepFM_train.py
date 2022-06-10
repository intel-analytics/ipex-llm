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
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy, AUC
from bigdl.friesian.feature import FeatureTable
from bigdl.dllib.utils.log4Error import *


spark_conf = {"spark.network.timeout": "10000000",
              "spark.sql.broadcastTimeout": "7200",
              "spark.sql.shuffle.partitions": "2000",
              "spark.locality.wait": "0s",
              "spark.sql.hive.filesourcePartitionFileCacheSize": "4096000000",
              "spark.sql.crossJoin.enabled": "true",
              "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
              "spark.kryo.unsafe": "true",
              "spark.kryoserializer.buffer.max": "1024m",
              "spark.task.cpus": "1",
              "spark.executor.heartbeatInterval": "200s",
              "spark.driver.maxResultSize": "40G",
              "spark.eventLog.enabled": "true",
              "spark.app.name": "recsys-2tower",
              "spark.executor.memoryOverhead": "120g"}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep FM Training')
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
    parser.add_argument('--epochs', default=1, type=int, help='train epoch')
    parser.add_argument('--batch_size', default=8000, type=int, help='batch size')
    parser.add_argument('--model_dir', default='snapshot', type=str,
                        help='snapshot directory name (default: snapshot)')
    parser.add_argument('--data_dir', type=str, help='data directory', required=True)
    parser.add_argument('--frequency_limit', type=int, default=25, help='frequency limit')
    args = parser.parse_args()

    if args.cluster_mode == "local":
        sc = init_orca_context("local", cores=args.executor_cores, memory=args.executor_memory)
    elif args.cluster_mode == "yarn":
        sc = init_orca_context("yarn-client", cores=args.executor_cores,
                               num_nodes=args.num_executor, memory=args.executor_memory,
                               driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                               conf=spark_conf, object_store_memory="40g", init_ray_on_spark=True,
                               extra_python_lib="model.py,evaluation.py")
    elif args.cluster_mode == "spark-submit":
        sc = init_orca_context("spark-submit", object_store_memory="40g")
    else:
        invalidInputError(False,
                          "cluster_mode should be one of 'local', 'yarn' and 'spark-submit',"
                          " but got " + args.cluster_mode)

    num_cols = ["enaging_user_follower_count", 'enaging_user_following_count',
                "engaged_with_user_follower_count", "engaged_with_user_following_count",
                "len_hashtags", "len_domains", "len_links"]
    cat_cols = ["engaged_with_user_is_verified", "enaging_user_is_verified", "present_media",
                "tweet_type", "language", 'present_media_language']
    embed_cols = ["enaging_user_id", "engaged_with_user_id", "hashtags", "present_links",
                  "present_domains"]
    all_features = embed_cols + cat_cols + num_cols

    train = FeatureTable.read_parquet(args.data_dir + "/train_parquet")
    test = FeatureTable.read_parquet(args.data_dir + "/test_parquet")

    test_user_ids = test.select("engaged_with_user_id").cast("engaged_with_user_id", "str").\
        to_list("engaged_with_user_id")
    test_labels = test.select("label").to_list("label")

    full = train.concat(test)
    reindex_tbls = full.gen_reindex_mapping(embed_cols, freq_limit=args.frequency_limit)
    full, min_max_dict = full.min_max_scale(num_cols)

    sparse_dims = {}
    for i, c, in enumerate(embed_cols):
        sparse_dims[c] = max(reindex_tbls[i].df.agg({c+"_new": "max"}).collect()[0]) + 1
    cat_dims = full.max(cat_cols).to_dict()
    cat_dims = dict(zip(cat_dims['column'], [dim + 1 for dim in cat_dims['max']]))
    sparse_dims.update(cat_dims)

    train = train.reindex(embed_cols, reindex_tbls)\
        .transform_min_max_scale(num_cols, min_max_dict)\
        .merge_cols(all_features, "feature") \
        .select(["label", "feature"])\
        .apply("label", "label", lambda x: [float(x)], dtype="array<float>")

    test = test.reindex(embed_cols, reindex_tbls) \
        .transform_min_max_scale(num_cols, min_max_dict) \
        .merge_cols(all_features, "feature") \
        .select(["label", "feature"]) \
        .apply("label", "label", lambda x: [float(x)], dtype="array<float>")
    test.cache()

    def model_creator(config):
        from deepctr_torch.inputs import SparseFeat, DenseFeat
        from model import DeepFM

        feature_columns = \
            [SparseFeat(feat, int(dim), 16) for feat, dim in config["sparse_dims"].items()] + \
            [DenseFeat(feat, 1) for feat in config["num_cols"]]
        model = DeepFM(linear_feature_columns=feature_columns,
                       dnn_feature_columns=feature_columns,
                       task='binary', l2_reg_embedding=1e-1)
        model.float()
        print(model)
        return model

    import torch

    def optim_creator(model, config):
        return torch.optim.Adam(model.parameters(), config['lr'])

    criterion = torch.nn.BCELoss()

    config = {'sparse_dims': sparse_dims, 'num_cols': num_cols, 'lr': args.lr}

    est = Estimator.from_torch(model=model_creator, optimizer=optim_creator, loss=criterion,
                               metrics=[Accuracy(), AUC()], use_tqdm=True, backend="ray",
                               config=config)
    train_stats = est.fit(data=train.df, feature_cols=["feature"], label_cols=["label"],
                          epochs=args.epochs, batch_size=args.batch_size)

    valid_stats = est.evaluate(data=test.df, feature_cols=["feature"], label_cols=["label"],
                               batch_size=args.batch_size)

    est.save(args.model_dir)
    print("Train stats: {}".format(train_stats))
    print("Validation stats: {}".format(valid_stats))

    predicts = est.predict(data=test.df, feature_cols=["feature"], batch_size=args.batch_size)
    predicts.show(10, False)

    est.shutdown()
    stop_orca_context()
