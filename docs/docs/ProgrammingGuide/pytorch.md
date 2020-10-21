Analytics-Zoo supports distributed Pytorch training and inferenceon on Apache Spark. User can
define their model and loss function with Pytorch API, and run it in a distributed environment
with the wrapper layers provided by Analytics Zoo.

# System Requirement
Pytorch version: 1.5.0 or above  
torchvision: 0.6.0 or above  
jep: 3.9.0  
Python: 3.7  

# Pytorch API

Two wrappers are defined in Analytics Zoo for Pytorch:

1. TorchModel: TorchModel is a wrapper class for Pytorch model.
User may create a TorchModel by providing a Pytorch model, e.g.
    ```python
    from zoo.pipeline.api.torch import TorchModel
    TorchModel.from_pytorch(torchvision.models.resnet18(pretrained=True))
    ```
The above line creates TorchModel wrapping a ResNet model, and user can use the TorchModel for
training or inference with Analytics Zoo.

2. TorchLoss: TorchLoss is a wrapper for loss functions defined by Pytorch.
User may create a TorchLoss from a Pytorch Criterion, 
    ```python
    from torch import nn
    from zoo.pipeline.api.torch import TorchLoss
    
    az_criterion = TorchLoss.from_pytorch(nn.MSELoss())
    ```
    or from a custom loss function, which takes input and label as parameters

    ```python
    from torch import nn
    from zoo.pipeline.api.torch import TorchLoss
    
    criterion = nn.MSELoss()

    # this loss function is calculating loss for a multi-output model
    def lossFunc(input, label):
        loss1 = criterion(input[0], label[0])
        loss2 = criterion(input[1], label[1])
        loss = loss1 + 0.4 * loss2
        return loss
    
    az_criterion = TorchLoss.from_pytorch(lossFunc)
    ```

# Examples
Here we provide a simple end to end example, where we use TorchModel and TorchLoss to
train a simple model with Spark DataFrame.
```python
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
from zoo.pipeline.api.torch import TorchModel, TorchLoss
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
    sparkConf = init_spark_conf().setAppName("example_pytorch").setMaster('local[1]')
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

    az_model = TorchModel.from_pytorch(torch_model)
    az_criterion = TorchLoss.from_pytorch(torch_criterion)

    classifier = NNClassifier(az_model, az_criterion) \
        .setBatchSize(4) \
        .setOptimMethod(Adam()) \
        .setLearningRate(0.01) \
        .setMaxEpoch(10)

    nnClassifierModel = classifier.fit(df)

    print("After training: ")
    res = nnClassifierModel.transform(df)
    res.show(10, False)

```
Please export `PYTHONHOME` env before you run this code, and we expects to see the output like:
```python
+---------+-----+----------+
|features |label|prediction|
+---------+-----+----------+
|[2.0,1.0]|1.0  |1.0       |
|[1.0,2.0]|0.0  |0.0       |
|[2.0,1.0]|1.0  |1.0       |
|[1.0,2.0]|0.0  |0.0       |
+---------+-----+----------+
```
