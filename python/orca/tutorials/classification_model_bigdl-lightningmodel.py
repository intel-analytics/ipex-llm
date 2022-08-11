import numpy as np
import pandas as pd

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
from pytorch_lightning.callbacks import Callback

# cluster_mode can be "local", "k8s" or "yarn"
sc = init_orca_context(cluster_mode="local", cores=4, memory="10g", num_nodes=1)

# data1
train = pd.read_csv('./train.csv', index_col='id')
train = train[~train.drop('target', axis=1).duplicated()]
train.to_csv('./train_fix.csv')

file_path = './train_fix.csv'
data_shard = bigdl.orca.data.pandas.read_csv(file_path)

def trans_func(df):
    df = df.rename(columns={'id': 'id0'})
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

# data2
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
file_path = './train.csv'
train = pd.read_csv(file_path, index_col='id')
train = train[~train.drop('target', axis=1).duplicated()]
X = pd.DataFrame(train.drop("target", axis=1))
lencoder = LabelEncoder()
y = pd.DataFrame(lencoder.fit_transform(train['target']), columns=['target'])
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_train, y_train = np.array(X_train, dtype=np.float32), y_train['target'].values
X_valid, y_valid = np.array(X_valid, dtype=np.float32), y_valid['target'].values

sc = init_orca_context(cluster_mode="yarn-client", cores=4, memory="10g", num_nodes=2)

class TPS05Dataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

train_dataset = TPS05Dataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
valid_dataset = TPS05Dataset(torch.from_numpy(X_valid).float(), torch.from_numpy(y_valid).long())

def train_loader_creator(config, batch_size):
    return DataLoader(dataset=train_dataset, batch_size=64)

def test_loader_creator(config, batch_size):
    return DataLoader(dataset=valid_dataset, batch_size=1)

# train
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
        return F2.cross_entropy(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        outputs = self.forward(x)
        loss = self.cross_entropy_loss(outputs, y)
        self.log('train_loss', loss)
        # return loss
        return {"loss": loss, "predictions": outputs, "labels": y}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        outputs = self.forward(x)
        loss = self.cross_entropy_loss(outputs, y)
        self.log('val_loss', loss)
        # return loss
        return {"loss": loss, "predictions": outputs, "labels": y}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        optimizer_one = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimizer_two = torch.optim.Adam(self.parameters(), lr=1e-3)
        # return optimizer
        return [{"optimizer": optimizer_one, "frequency": 5}, {"optimizer": optimizer_two, "frequency": 10}]
        # return [optimizer], [lr_scheduler]


class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")

def model_creator(config):
    model = LightningMNISTClassifier()
    return model

est = Estimator.from_torch(model=model_creator, metrics=[Accuracy()], backend="ray")


est.fit(data=shards_train1, feature_cols=['x_scaled'], label_cols=['target'], validation_data=shards_val1,
        epochs=1, batch_size=BATCH_SIZE, callbacks=[MyPrintingCallback()])

# est.fit(data=train_loader_creator, validation_data=test_loader_creator,
#         epochs=1, batch_size=BATCH_SIZE, callbacks=[MyPrintingCallback()])

result = est.evaluate(data=shards_val1, feature_cols=['x_scaled'], label_cols=['target'], batch_size=1)

# result = est.evaluate(data=test_loader_creator, batch_size=1)

for r in result:
    print(r, ":", result[r])
