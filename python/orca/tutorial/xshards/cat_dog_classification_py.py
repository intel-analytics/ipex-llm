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
# This example is adapted from
# https://www.kaggle.com/code/reukki/pytorch-cnn-tutorial-with-cats-and-dogs

from bigdl.orca.data.shard import SparkXShards
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy
import bigdl.orca.data.pandas
from torch import nn
import torch
from torchvision import transforms
import torch.optim as optim

from bigdl.orca import init_orca_context, stop_orca_context
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
import bigdl.orca.data.image
from pyspark import SparkConf, SparkContext

conf = {
    "spark.app.name": "myapp",
    "spark.local.dir": "/tmp",
    'spark.executorEnv.ARROW_LIBHDFS_DIR': '/opt/cloudera/parcels/CDH/lib64'}

sc = init_orca_context(cluster_mode="local", cores=4, memory="4g", conf=conf)
path = '/Users/guoqiong/intelWork/data/dogs-vs-cats/small/'


# executor_memory='40g'
# executor_cores=20
# driver_memory='10g'
# driver_cores=20
# num_executor=3
#
# sc = init_orca_context("yarn-client", cores=executor_cores,
#                   num_nodes=num_executor, memory=executor_memory,
#                   driver_cores=driver_cores, driver_memory=driver_memory)
#
# path = 'hdfs://172.16.0.105:8020/dogs-vs-cats/small/'


def get_label(file_name):
    label = [1] if 'dog' in file_name.split('/')[-1] else [0]
    return label


data_shard = bigdl.orca.data.image.read_images_spark(path, get_label)

data_shard = bigdl.orca.data.image.read_images_pil(path)
print(data_shard.collect()[0])
# import sys
# sys.exit()

def train_transform(im):
    features = im[0]
    features = transforms.Resize((224, 224))(features)
    features = transforms.RandomResizedCrop(224)(features)
    features = transforms.RandomHorizontalFlip()(features)
    features = transforms.ToTensor()(features)
    features = features.numpy()
    return features, im[1]

to_ndarray = lambda x: {'x': np.array([x[0]]).astype(np.float32),
                        'y': np.array([x[1]]).astype(np.float32)}

data_shard = data_shard.transform_shard(train_transform)
# data_shard = data_shard.transform_shard(to_ndarray)
# print(data_shard.collect()[0])
print("******************")
data_shard = data_shard.stack_feature_labels()

# data_shard = data_shard.transform_shard(to_ndarray)
#

def train_transform(im):
    features = im[0]
    features = transforms.Resize((224, 224))(features)
    features = transforms.RandomResizedCrop(224)(features)
    features = transforms.RandomHorizontalFlip()(features)
    features = transforms.ToTensor()(features)
    features = features.numpy()
    return {'x': np.array([features]).astype(np.float32),
            'y': np.array([im[1]]).astype(np.float32)}


# data_shard = data_shard.transform_shard(train_transform)

print("******************")
print(data_shard.collect()[0]['x'].shape)
print(data_shard.collect()[0]['y'].shape)


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(3 * 3 * 64, 10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.act = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        out = self.act(out)
        return out


criterion = nn.BCELoss()


def model_creator(config):
    model = Cnn().to(device)
    model.train()
    return model


def optimizer_creator(model, config):
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    return optimizer


orca_estimator = Estimator.from_torch(model=model_creator,
                                      optimizer=optimizer_creator,
                                      loss=criterion,
                                      metrics=[Accuracy()],
                                      backend="spark")

orca_estimator.fit(data=data_shard, epochs=10, batch_size=32)

stop_orca_context()
