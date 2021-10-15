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
from unittest import TestCase

import os
import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.data.pandas import read_csv
from bigdl.orca.data import SparkXShards
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy
from bigdl.orca.learn.trigger import EveryEpoch
from bigdl.orca.learn.optimizers import SGD
from bigdl.orca.learn.optimizers.schedule import Default
from bigdl.orca import OrcaContext
import tempfile

resource_path = os.path.join(os.path.split(__file__)[0], "../../resources")


class TestEstimatorForSpark(TestCase):

    def setUp(self):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        self.sc = init_orca_context(cores=4)

    def tearDown(self):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        stop_orca_context()

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
                "x": np.stack([df['user'].to_numpy(), df['item'].to_numpy()], axis=1),
                "y": df['label'].to_numpy()
            }
            return result

        def transform_del_y(d):
            result = {"x": d["x"]}
            return result

        OrcaContext.pandas_read_backend = "pandas"
        file_path = os.path.join(resource_path, "orca/learn/ncf.csv")
        data_shard = read_csv(file_path)
        data_shard = data_shard.transform_shard(transform)

        with tempfile.TemporaryDirectory() as temp_dir_name:
            estimator = Estimator.from_torch(model=model, loss=loss_func,
                                             metrics=[Accuracy()],
                                             optimizer=SGD(learningrate_schedule=Default()),
                                             model_dir=temp_dir_name)
            estimator.fit(data=data_shard, epochs=4, batch_size=2, validation_data=data_shard,
                          checkpoint_trigger=EveryEpoch())
            state_dict1 = estimator.get_model().state_dict()

            estimator.evaluate(data_shard, batch_size=2)
            est2 = Estimator.from_torch(model=model, loss=loss_func,
                                        metrics=[Accuracy()],
                                        optimizer=None)
            est2.load_orca_checkpoint(temp_dir_name)
            state_dict2 = est2.get_model().state_dict()

            for name in state_dict1:
                para1 = state_dict1[name]
                para2 = state_dict2[name]
                assert torch.all(torch.eq(para1, para2)), "After reloading the model, " \
                                                          "%r does not match" % name

            est2.fit(data=data_shard, epochs=8, batch_size=2, validation_data=data_shard,
                     checkpoint_trigger=EveryEpoch())
            est2.evaluate(data_shard, batch_size=2)
            pred_result = est2.predict(data_shard)
            pred_c = pred_result.collect()
            assert(pred_result, SparkXShards)
            pred_shard = data_shard.transform_shard(transform_del_y)
            pred_result2 = est2.predict(pred_shard)
            pred_c_2 = pred_result2.collect()
            assert (pred_c[0]["prediction"] == pred_c_2[0]["prediction"]).all()

    def test_bigdl_pytorch_estimator_pandas_dataframe(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc = nn.Linear(1, 10)

            def forward(self, x):
                x = torch.unsqueeze(x, dim=1)
                x = self.fc(x)
                return F.log_softmax(x, dim=1)

        def loss_func(input, target):
            return nn.CrossEntropyLoss().forward(input, target.flatten().long())
        model = SimpleModel()

        OrcaContext.pandas_read_backend = "pandas"
        file_path = os.path.join(resource_path, "orca/learn/simple_feature_label.csv")
        data_shard = read_csv(file_path)

        with tempfile.TemporaryDirectory() as temp_dir_name:
            estimator = Estimator.from_torch(model=model, loss=loss_func,
                                             metrics=[Accuracy()],
                                             optimizer=SGD(learningrate_schedule=Default()),
                                             model_dir=temp_dir_name)
            estimator.fit(data=data_shard, epochs=1, batch_size=4, feature_cols=['feature'],
                          label_cols=['label'], validation_data=data_shard,
                          checkpoint_trigger=EveryEpoch())
            estimator.evaluate(data_shard, batch_size=4, feature_cols=['feature'],
                               label_cols=['label'])
            est2 = Estimator.from_torch(model=model, loss=loss_func,
                                        metrics=[Accuracy()],
                                        optimizer=None)
            est2.load_orca_checkpoint(temp_dir_name)
            est2.predict(data_shard, batch_size=4, feature_cols=['feature'])

if __name__ == "__main__":
    pytest.main([__file__])
