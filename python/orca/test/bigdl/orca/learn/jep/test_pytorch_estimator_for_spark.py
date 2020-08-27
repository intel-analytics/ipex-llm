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
from unittest import TestCase

import pytest
import torch.nn as nn
import torch.nn.functional as F

from zoo.orca.data.pandas import read_csv
from zoo.orca.learn.pytorch import Estimator
from zoo.orca.learn.metrics import Accuracy
from zoo.orca.learn.trigger import EveryEpoch
from zoo.common.nncontext import *
from bigdl.optim.optimizer import SGD
from zoo.orca import OrcaContext

resource_path = os.path.join(os.path.split(__file__)[0], "../../../resources")


class TestEstimatorForSpark(TestCase):

    def setUp(self):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        self.sc = init_spark_on_local(4)

    def tearDown(self):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        self.sc.stop()

    def test_bigdl_pytorch_estimator_shard(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc = nn.Linear(2, 2)

            def forward(self, x):
                x = self.fc(x)
                return F.log_softmax(x, dim=1)

        model = SimpleModel()

        def loss_func(input, target):
            return nn.CrossEntropyLoss().forward(input, target.flatten().long())

        def transform(df):
            result = {
                "x": [df['user'].to_numpy(), df['item'].to_numpy()],
                "y": df['label'].to_numpy()
            }
            return result

        OrcaContext.pandas_read_backend = "pandas"
        file_path = os.path.join(resource_path, "orca/learn/ncf.csv")
        data_shard = read_csv(file_path)
        data_shard = data_shard.transform_shard(transform)

        estimator = Estimator.from_torch(model=model, loss=loss_func,
                                         optimizer=SGD(), backend="bigdl")
        estimator.fit(data=data_shard, epochs=4, batch_size=2, validation_data=data_shard,
                      validation_methods=[Accuracy()], checkpoint_trigger=EveryEpoch())
        estimator.evaluate(data_shard, validation_methods=[Accuracy()], batch_size=2)


if __name__ == "__main__":
    pytest.main([__file__])
