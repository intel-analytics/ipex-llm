Analytics-Zoo supports distributed Pytorch training and inferenceon on Apache Spark. User can
define their model and loss function with Pytorch API, and run it in a distributed environment
with the wrapper layers provided by Analytics Zoo.

# System Requirement
Pytorch version: 1.1.0
torchvision: 2.2.0

tested OS version (all 64-bit): __Ubuntu 16.04 or later__ . We expect it to 
support a wide range of Operating Systems, yet other systems have not been fully tested with.
Please create issues on [issue page](https://github.com/intel-analytics/analytics-zoo/issues)
if any error is found.


# Pytorch API

Two wrappers are defined in Analytics Zoo for Pytorch:

1. TorchNet: TorchNet is a wrapper class for Pytorch model.
User may create a TorchNet by providing a Pytorch model and example input or expected size, e.g.
```python
    from zoo.pipeline.api.net.torch_net import TorchNet
    TorchNet.from_pytorch(torchvision.models.resnet18(pretrained=True).eval(), [1, 3, 224, 224])
```
The above line creates TorchNet wrapping a ResNet model, and user can use the TorchNet for
training or inference with Analytics Zoo. Internally, we create a sample input
from the input_shape provided, and use torch script module to trace the tensor operations
performed on the input sample. The result TorchNet extends from BigDL module, and can be used
with local or distributed data (RDD or DataFrame) just like other layers. For multi-input
models, please use tuple of tensors or tuple of expected tensor sizes as example input.

2. TorchCriterion: TorchCriterion is a wrapper for loss functions defined by Pytorch.
User may create a TorchCriterion from a Pytorch Criterion, 
```python
    from torch import nn
    from zoo.pipeline.api.net.torch_criterion import TorchCriterion
    
    az_criterion = TorchCriterion.from_pytorch(loss=nn.MSELoss(),
                                               input=[1, 1],
                                               label=[1, 1])
```
or from a custom loss function, which takes input and label as parameters

```python
    from torch import nn
    from zoo.pipeline.api.net.torch_criterion import TorchCriterion
    
    criterion = nn.MSELoss()

    # this loss function is calculating loss for a multi-output model
    def lossFunc(input, label):
        loss1 = criterion(input[0], label[0])
        loss2 = criterion(input[1], label[1])
        loss = loss1 + 0.4 * loss2
        return loss
    
    az_criterion = TorchCriterion.from_pytorch(loss=lossFunc,
                                               input=(torch.ones(2, 2), torch.ones(2, 1)),
                                               label=(torch.ones(2, 2), torch.ones(2, 1)))
```
Similar to TorchNet, we also need users to provide example input shape or example input data,
to trace the operations in the loss functions. The created TorchCriterion extends BigDL
criterion, and can be used similarly as other criterions.

# Examples
Here we provide a simple end to end example, where we use TorchNet and TorchCriterion to
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

```

and we expects to see the output like:
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

More Pytorch examples (ResNet, Lenet etc.) are available [here](../../../pyzoo/zoo/examples/pytorch).

