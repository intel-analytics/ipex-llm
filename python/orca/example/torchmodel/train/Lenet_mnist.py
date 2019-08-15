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
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from bigdl.optim.optimizer import *
from torchvision import datasets
from zoo.common.nncontext import *
from zoo.pipeline.nnframes import *
from zoo.pipeline.api.net.torch_net import TorchNet
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
from zoo.pipeline.api.net.torch_criterion import TorchCriterion


# define model with Pytorch API
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    sparkConf = init_spark_conf().setAppName("test_pytorch_lenet")
    sc = init_nncontext(sparkConf)
    spark = SparkSession.builder.config(conf=sparkConf).getOrCreate()

    mnist = datasets.MNIST('../data', train=True, download=True)
    X_train = mnist.data.numpy() / 255.0
    Y_train = mnist.train_labels.float().numpy()
    pd_df = pd.DataFrame()
    pd_df['features'] = X_train.reshape((X_train.shape[0], 784)).tolist()
    pd_df['label'] = Y_train.reshape((Y_train.shape[0])).tolist()

    mnistDF = spark.createDataFrame(pd_df)
    (trainingDF, validationDF) = mnistDF.randomSplit([0.8, 0.2])
    trainingDF.show()

    # define loss with Pytorch API
    def lossFunc(input, target):
        return nn.CrossEntropyLoss().forward(input, target.flatten().long())

    torch_model = LeNet()
    model = TorchNet.from_pytorch(torch_model, [1, 1, 28, 28])
    criterion = TorchCriterion.from_pytorch(lossFunc, [1, 10], torch.LongTensor([5]))
    classifier = NNClassifier(model, criterion, SeqToTensor([1, 28, 28])) \
        .setBatchSize(64) \
        .setOptimMethod(Adam()) \
        .setLearningRate(0.001)\
        .setMaxEpoch(2)

    nnClassifierModel = classifier.fit(trainingDF)

    print("After training: ")
    shift = udf(lambda p: p - 1, DoubleType())
    res = nnClassifierModel.transform(validationDF) \
        .withColumn("prediction", shift(col('prediction')))
    res.show(100)

    correct = res.filter("label=prediction").count()
    overall = res.count()
    accuracy = correct * 1.0 / overall
    print("Validation accuracy = %g " % accuracy)
