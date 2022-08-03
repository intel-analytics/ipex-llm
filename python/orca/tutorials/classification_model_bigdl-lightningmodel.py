# We have to prepare for this journey .... import modules is e great idea .... :)
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F2

from bigdl.orca import init_orca_context, OrcaContext
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy

import bigdl.orca.data
import bigdl.orca.data.pandas
from bigdl.orca.data import SharedValue
from bigdl.orca.data import SparkXShards

from bigdl.orca.data.transformer import *

import pytorch_lightning as pl


# cluster_mode can be "local", "k8s" or "yarn"
sc = init_orca_context(cluster_mode="local", cores=4, memory="10g", num_nodes=1)

train = pd.read_csv('./train.csv', index_col = 'id')
train = train[~train.drop('target', axis = 1).duplicated()]
train.to_csv('./train_fix.csv')

file_path = './train_fix.csv'
data_shard = bigdl.orca.data.pandas.read_csv(file_path)

def trans_func(df):
    df = df.rename(columns={'id':'id0'})
    return df
data_shard = data_shard.transform_shard(trans_func)

scale = StringIndexer(inputCol='target')
transformed_data_shard = scale.fit_transform(data_shard)

def trans_func(df):
    df['target'] = df['target']-1
    return df
transformed_data_shard = transformed_data_shard.transform_shard(trans_func)

RANDOM_STATE = 2021
def split_train_test(data):
    train, test = train_test_split(data, test_size=0.2, random_state=RANDOM_STATE)
    return train, test

shards_train, shards_val = transformed_data_shard.transform_shard(split_train_test).split()

feature_list = []
for i in range(50):
    feature_list.append('feature_' + str(i))
scale = MinMaxScaler(inputCol=feature_list, outputCol="x_scaled")
shards_train = scale.fit_transform(shards_train)
shards_val = scale.transform(shards_val)

def trans_func(df):
    df['x_scaled'] = df['x_scaled'].apply(lambda x:np.array(x,dtype=np.float32))
    df['target'] = df['target'].apply(lambda x:np.long(x))
    return df
shards_train1 = shards_train.transform_shard(trans_func)
shards_val1 = shards_val.transform_shard(trans_func)

torch.manual_seed(0)
BATCH_SIZE = 64
NUM_CLASSES = 4
NUM_EPOCHS = 100


def linear_block(in_features, out_features, p_drop, *args, **kwargs):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(),
        nn.Dropout(p=p_drop)
    )


# class TPS05ClassificationSeq(nn.Module):
#     def __init__(self):
#         super(TPS05ClassificationSeq, self).__init__()
#         num_feature = len(train.columns) - 1
#         num_class = 4
#         self.linear = nn.Sequential(
#             linear_block(num_feature, 100, 0.3),
#             linear_block(100, 250, 0.3),
#             linear_block(250, 128, 0.3),
#         )
#
#         self.out = nn.Sequential(
#             nn.Linear(128, num_class)
#         )
#
#     def forward(self, x):
#         x = self.linear(x)
#         return self.out(x)


class LightningMNISTClassifier(pl.LightningModule):
    def __init__(self):
        super(LightningMNISTClassifier, self).__init__()
        num_feature = len(train.columns) - 1
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

    def cross_entropy_loss(self, logits, labels):
        # print(type(F), "+++++++++++++++++++++++++++")
        return F2.cross_entropy(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def model_creator(config):
    model = LightningMNISTClassifier()
    return model

# def model_creator2(config):
#     model = TPS05ClassificationSeq()
#     return model

# def optim_creator(model, config):
#     return optim.Adam(model.parameters(), lr=0.001)

# criterion = nn.CrossEntropyLoss()

# est = Estimator.from_torch(model=model_creator2, optimizer=optim_creator, loss=criterion, metrics=[Accuracy()], backend="ray")

est = Estimator.from_torch(model=model_creator, metrics=[Accuracy()], backend="ray")


est.fit(data=shards_train1, feature_cols=['x_scaled'], label_cols=['target'], validation_data=shards_val1,
        epochs=1, batch_size=BATCH_SIZE)

result = est.evaluate(data=shards_val1, feature_cols=['x_scaled'], label_cols=['target'], batch_size=1)

for r in result:
    print(r, ":", result[r])
