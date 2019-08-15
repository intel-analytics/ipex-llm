#
# Copyright 2018 Analytics Zoo Authors.
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
import torch
import torch.nn as nn
from bigdl.optim.optimizer import Adam
from zoo.common.nncontext import *
from zoo.pipeline.api.net.torch_net import TorchNet
from zoo.pipeline.api.net.torch_criterion import TorchCriterion
from zoo.pipeline.nnframes import *

from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession


# define model with Pytorch
class SimpleTorchModel(nn.Module):
    def __init__(self):
        super(SimpleTorchModel, self).__init__()
        self.dense1 = nn.Linear(2, 4)
        self.dense2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.dense1(x)
        x = torch.sigmoid(self.dense2(x))
        return x

if __name__ == '__main__':
    sparkConf = init_spark_conf().setAppName("testNNClassifer").setMaster('local[1]')
    sc = init_nncontext(sparkConf)
    spark = SparkSession \
        .builder \
        .getOrCreate()

    df = spark.createDataFrame(
        [(Vectors.dense([2.0, 1.0]), 1.0),
         (Vectors.dense([1.0, 2.0]), 0.0),
         (Vectors.dense([2.0, 1.0]), 1.0),
         (Vectors.dense([1.0, 2.0]), 0.0)],
        ["features", "label"])

    torch_model = SimpleTorchModel()
    torch_criterion = nn.MSELoss()

    az_model = TorchNet.from_pytorch(torch_model, [1, 2])
    az_criterion = TorchCriterion.from_pytorch(torch_criterion, [1, 1], [1, 1])

    classifier = NNClassifier(az_model, az_criterion) \
        .setBatchSize(4) \
        .setOptimMethod(Adam()) \
        .setLearningRate(0.01) \
        .setMaxEpoch(10)

    nnClassifierModel = classifier.fit(df)

    print("After training: ")
    res = nnClassifierModel.transform(df)
    res.show(10, False)
