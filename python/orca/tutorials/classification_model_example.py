import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from bigdl.orca import init_orca_context
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy
import bigdl.orca.data
import bigdl.orca.data.pandas
from bigdl.orca.data.transformer import *
from optparse import OptionParser
import sys


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-f", type=str, dest="file_path", help="The file path to be read")
    parser.add_option("--deploy-mode", type=str, dest="deployMode", default="local", help="deploy mode, local, spark-submit, yarn-client or yarn-cluster")
    (options, args) = parser.parse_args(sys.argv)

    file_path = options.file_path
    # cluster_mode can be "local", "k8s" or "yarn"
    sc = init_orca_context(cluster_mode=options.deployMode, cores=4, memory="10g", num_nodes=1)

    # Load data
    data_shard = bigdl.orca.data.pandas.read_csv(file_path)
    # Duplicate the dataframe
    data_shard = data_shard.deduplicates()
    # Labelencode y
    def trans_func(df):
        df = df.rename(columns={'id': 'id0'})
        return df
    data_shard = data_shard.transform_shard(trans_func)
    scale = StringIndexer(inputCol='target')
    transformed_data_shard = scale.fit_transform(data_shard)
    def trans_func(df):
        df['target'] = df['target'] - 1
        return df
    transformed_data_shard = transformed_data_shard.transform_shard(trans_func)

    # Split train and test set
    RANDOM_STATE = 2021
    def split_train_test(data):
        train, test = train_test_split(data, test_size=0.2, random_state=RANDOM_STATE)
        return train, test
    shards_train, shards_val = transformed_data_shard.transform_shard(split_train_test).split()

    # Transform the feature columns
    feature_list = []
    for i in range(50):
        feature_list.append('feature_' + str(i))
    scale = MinMaxScaler(inputCol=feature_list, outputCol="x_scaled")
    shards_train = scale.fit_transform(shards_train)
    shards_val = scale.transform(shards_val)

    # Change data types
    def trans_func(df):
        df['x_scaled'] = df['x_scaled'].apply(lambda x: np.array(x, dtype=np.float32))
        df['target'] = df['target'].apply(lambda x: np.long(x))
        return df
    shards_train1 = shards_train.transform_shard(trans_func)
    shards_val1 = shards_val.transform_shard(trans_func)

    # Model
    torch.manual_seed(0)
    BATCH_SIZE = 64
    NUM_CLASSES = 4
    NUM_EPOCHS = 100
    NUM_FEATURES = 50


    def linear_block(in_features, out_features, p_drop, *args, **kwargs):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(p=p_drop)
        )


    class TPS05ClassificationSeq(nn.Module):
        def __init__(self):
            super(TPS05ClassificationSeq, self).__init__()
            num_feature = NUM_FEATURES
            num_class = 4
            self.linear = nn.Sequential(
                linear_block(num_feature, 100, 0.3),
                linear_block(100, 250, 0.3),
                linear_block(250, 128, 0.3),
            )

            self.out = nn.Sequential(
                nn.Linear(128, num_class)
            )

        def forward(self, x):
            x = self.linear(x)
            return self.out(x)


    def model_creator(config):
        model = TPS05ClassificationSeq()
        return model

    def optim_creator(model, config):
        return optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()

    est = Estimator.from_torch(model=model_creator, optimizer=optim_creator, loss=criterion, metrics=[Accuracy()],
                               backend="ray")

    est.fit(data=shards_train1, feature_cols=['x_scaled'], label_cols=['target'], validation_data=shards_val1, epochs=1,
            batch_size=BATCH_SIZE)

    result = est.evaluate(data=shards_val1, feature_cols=['x_scaled'], label_cols=['target'], batch_size=1)

    for r in result:
        print(r, ":", result[r])