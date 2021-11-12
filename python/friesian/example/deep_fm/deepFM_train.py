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

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models.deepfm import *
from deepctr_torch.models.basemodel import *
from evaluation import uAUC
from bigdl.friesian.feature import FeatureTable
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy
import argparse

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


class DeepFM(BaseModel):
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, use_fm=True,
                 dnn_hidden_units=(512, 256, 128), l2_reg_linear=0.00001, l2_reg_embedding=0.00001,
                 l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu',
                 dnn_use_bn=False, task='binary', device='cpu', gpus=None):

        super(DeepFM, self).__init__(linear_feature_columns, dnn_feature_columns,
                                     l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding,
                                     init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)

        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        if use_fm:
            self.fm = FM()

        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                           use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)

            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0],
                       self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self\
            .input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        logit = self.linear_model(X)

        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            logit += self.fm(fm_input)

        if self.use_dnn:
            dnn_input = combined_dnn_input(
                sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit

        y_pred = self.out(logit)

        return y_pred


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
    parser.add_argument('--data_dir', type=str, help='data directory')
    parser.add_argument('--frequency_limit', type=int, default=25, help='frequency limit')

    args = parser.parse_args()
    if args.cluster_mode == "local":
        sc = init_orca_context("local", init_ray_on_spark=True)
    elif args.cluster_mode == "standalone":
        sc = init_orca_context("standalone", master=args.master,
                               cores=args.executor_cores, num_nodes=args.num_executor,
                               memory=args.executor_memory,
                               driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                               conf=spark_conf,
                               init_ray_on_spark=True)
    elif args.cluster_mode == "yarn":
        sc = init_orca_context("yarn-client", cores=args.executor_cores,
                               num_nodes=args.num_executor, memory=args.executor_memory,
                               driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                               conf=spark_conf, object_store_memory="80g", init_ray_on_spark=True)
    elif args.cluster_mode == "spark-submit":
        sc = init_orca_context("spark-submit")

    num_cols = ["enaging_user_follower_count", 'enaging_user_following_count',
                "engaged_with_user_follower_count", "engaged_with_user_following_count",
                "len_hashtags", "len_domains", "len_links", "hashtags", "present_links",
                "present_domains"]
    cat_cols = ["engaged_with_user_is_verified", "enaging_user_is_verified",
                "present_media", "tweet_type", "language"]
    embed_cols = ["enaging_user_id", "engaged_with_user_id", "hashtags", "present_links",
                  "present_domains"]

    train = FeatureTable.read_parquet(args.data_dir + "/train_parquet")
    test = FeatureTable.read_parquet(args.data_dir + "/test_parquet")

    test_user_ids = test.select("engaged_with_user_id").cast("engaged_with_user_id", "str").\
        to_list("engaged_with_user_id")
    test_labels = test.select("label").to_list("label")

    full = train.concat(test, "outer")
    sparse_dims = full.max(embed_cols).to_dict()
    sparse_dims = dict(zip(sparse_dims['column'], [dim + 1 for dim in sparse_dims['max']]))

    fixlen_feature_columns = [SparseFeat(feat, int(sparse_dims[feat])) for feat in sparse_dims] + \
                             [DenseFeat(feat, 1, ) for feat in num_cols]
    feature_names = get_feature_names(fixlen_feature_columns)
    reindex_tbls = full.gen_reindex_mapping(embed_cols, freq_limit=args.frequency_limit)
    full, min_max_dict = full.min_max_scale(num_cols)

    train = train.reindex(embed_cols, reindex_tbls)\
        .transform_min_max_scale(num_cols, min_max_dict)\
        .merge_cols(feature_names, "combine") \
        .select(["label", "combine"])\
        .apply("label", "label", lambda x: [float(x)], dtype="array<float>")

    test = test.reindex(embed_cols, reindex_tbls) \
        .transform_min_max_scale(num_cols, min_max_dict) \
        .merge_cols(feature_names, "combine") \
        .select(["label", "combine"]) \
        .apply("label", "label", lambda x: [float(x)], dtype="array<float>")

    config = {'linear_feature_columns': fixlen_feature_columns,
              'dnn_feature_columns': fixlen_feature_columns, 'feature_names': feature_names,
              'lr': args.lr}

    def model_creator(config):
        model = DeepFM(linear_feature_columns=config["linear_feature_columns"],
                       dnn_feature_columns=config["dnn_feature_columns"],
                       task='binary', l2_reg_embedding=1e-1)
        model.float()
        return model

    def optim_creator(model, config):
        return torch.optim.Adam(model.parameters(), config['lr'])

    criterion = torch.nn.BCELoss()

    est = Estimator.from_torch(model=model_creator, optimizer=optim_creator, loss=criterion,
                               metrics=[Accuracy()],
                               backend="torch_distributed", config=config)
    train_stats = est.fit(data=train.df, feature_cols=["combine"], label_cols=["label"],
                          epochs=args.epochs,
                          batch_size=args.batch_size)
    valid_stats = est.evaluate(data=test.df, feature_cols=["combine"], label_cols=["label"],
                               batch_size=args.batch_size)

    predicts = est.predict(data=test.df, feature_cols=["combine"], batch_size=args.batch_size)
    predicts = predicts.select("prediction").rdd.flatMap(lambda x: x).collect()
    auc = uAUC(test_labels, predicts, test_user_ids)
    print("Train stats: {}".format(train_stats))
    print("Validation stats: {}".format(valid_stats))
    print("AUC: ", auc)
    est.shutdown()
    stop_orca_context()
