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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.sql import SparkSession
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


class TestEstimatorForDataFrame(TestCase):
    def setUp(self):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        self.sc = init_orca_context(cores=4)

        def to_array_(v):
            return v.toArray().tolist()

        def flatten_(v):
            result = []
            for elem in v:
                result.extend(elem.toArray().tolist())
            return result

        self.spark = SparkSession(self.sc)
        self.spark.udf.register("to_array", to_array_, ArrayType(DoubleType()))
        self.spark.udf.register("flatten", flatten_, ArrayType(DoubleType()))

    def tearDown(self):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        stop_orca_context()

    def test_bigdl_pytorch_estimator_dataframe_predict(self):
        def loss_func(input, target):
            return nn.CrossEntropyLoss().forward(input, target.flatten().long())

        class IdentityNet(nn.Module):
            def __init__(self):
                super().__init__()
                # need this line to avoid optimizer raise empty variable list
                self.fc1 = nn.Linear(5, 5)

            def forward(self, input_):
                return input_

        model = IdentityNet()
        rdd = self.sc.range(0, 100)
        df = rdd.map(lambda x: ([float(x)] * 5,
                                [int(np.random.randint(0, 2,
                                                       size=()))])).toDF(["feature", "label"])

        with tempfile.TemporaryDirectory() as temp_dir_name:
            estimator = Estimator.from_torch(model=model, loss=loss_func,
                                             optimizer=SGD(learningrate_schedule=Default()),
                                             model_dir=temp_dir_name)
            result = estimator.predict(df, feature_cols=["feature"])
            expr = "sum(cast(feature <> to_array(prediction) as int)) as error"
            assert result.selectExpr(expr).first()["error"] == 0

    def test_bigdl_pytorch_estimator_dataframe_fit_evaluate(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc = nn.Linear(5, 5)

            def forward(self, x):
                x = self.fc(x)
                return F.log_softmax(x, dim=1)

        model = SimpleModel()

        def loss_func(input, target):
            return nn.CrossEntropyLoss().forward(input, target.flatten().long())

        rdd = self.sc.range(0, 100)
        df = rdd.map(lambda x: ([float(x)] * 5,
                                [int(np.random.randint(0, 2,
                                                       size=()))])).toDF(["feature", "label"])

        with tempfile.TemporaryDirectory() as temp_dir_name:
            estimator = Estimator.from_torch(model=model, loss=loss_func, metrics=[Accuracy()],
                                             optimizer=SGD(learningrate_schedule=Default()),
                                             model_dir=temp_dir_name)
            estimator.fit(data=df, epochs=4, batch_size=2, validation_data=df,
                          checkpoint_trigger=EveryEpoch(),
                          feature_cols=["feature"], label_cols=["label"])
            eval_result = estimator.evaluate(df, batch_size=2,
                                             feature_cols=["feature"], label_cols=["label"])
            assert isinstance(eval_result, dict)


if __name__ == "__main__":
    pytest.main([__file__])
