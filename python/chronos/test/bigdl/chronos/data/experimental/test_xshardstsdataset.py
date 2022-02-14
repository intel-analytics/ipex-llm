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

import pytest
import numpy as np
import pandas as pd
import random
import os

from unittest import TestCase
from bigdl.chronos.data import TSDataset
from bigdl.chronos.data.experimental import XShardsTSDataset
from bigdl.orca.data.pandas import read_csv
from bigdl.orca.common import init_orca_context, stop_orca_context

from pandas.testing import assert_frame_equal
from numpy.testing import assert_array_almost_equal


class TestXShardsTSDataset(TestCase):

    def setUp(self):
        self.resource_path = os.path.join(os.path.split(__file__)[0], "../../resources/")

    def tearDown(self):
        pass
    
    @classmethod
    def tearDownClass(cls):
        # stop possible active_spark_context
        from pyspark import SparkContext
        from bigdl.orca.ray import RayContext
        if SparkContext._active_spark_context is not None:
            print("Stopping spark_orca context")
            sc = SparkContext.getOrCreate()
            if sc.getConf().get("spark.master").startswith("spark://"):
                from bigdl.dllib.nncontext import stop_spark_standalone
                stop_spark_standalone()
            sc.stop()

    def test_xshardstsdataset_initialization(self):
        from bigdl.orca import init_orca_context, stop_orca_context
        init_orca_context(cores=4, memory="2g")
        shards_single = read_csv(os.path.join(self.resource_path, "single.csv"))
        tsdata = XShardsTSDataset.from_xshards(shards_single, dt_col="datetime", target_col="value",
                                               extra_feature_col=["extra feature"], id_col="id")
        assert tsdata._id_list == [0]
        assert tsdata.feature_col == ["extra feature"]
        assert tsdata.target_col == ["value"]
        assert tsdata.dt_col == "datetime"
        assert tsdata.shards.num_partitions() == 1

        tsdata = XShardsTSDataset.from_xshards(shards_single, dt_col="datetime",
                                               target_col=["value"],
                                               extra_feature_col="extra feature", id_col="id")
        assert tsdata._id_list == [0]
        assert tsdata.feature_col == ["extra feature"]
        assert tsdata.target_col == ["value"]
        assert tsdata.dt_col == "datetime"
        assert tsdata.shards.num_partitions() == 1

        tsdata = XShardsTSDataset.from_xshards(shards_single, dt_col="datetime",
                                               target_col=["value"],
                                               extra_feature_col="extra feature")
        assert tsdata._id_list == ["0"]
        assert tsdata.feature_col == ["extra feature"]
        assert tsdata.target_col == ["value"]
        assert tsdata.dt_col == "datetime"
        assert tsdata.shards.num_partitions() == 1
        # stop_orca_context()

    def test_xshardstsdataset_initialization_multiple(self):
        # from bigdl.orca import init_orca_context, stop_orca_context
        # init_orca_context(cores=4, memory="2g")
        shards_multiple = read_csv(os.path.join(self.resource_path, "multiple.csv"))
        # legal input
        tsdata = XShardsTSDataset.from_xshards(shards_multiple, dt_col="datetime",
                                               target_col="value",
                                               extra_feature_col=["extra feature"], id_col="id")
        assert tsdata._id_list == [0, 1]
        assert tsdata.feature_col == ["extra feature"]
        assert tsdata.target_col == ["value"]
        assert tsdata.dt_col == "datetime"
        assert tsdata.shards.num_partitions() == 2

        tsdata = XShardsTSDataset.from_xshards(shards_multiple, dt_col="datetime",
                                               target_col=["value"],
                                               extra_feature_col="extra feature", id_col="id")
        assert tsdata._id_list == [0, 1]
        assert tsdata.feature_col == ["extra feature"]
        assert tsdata.target_col == ["value"]
        assert tsdata.dt_col == "datetime"
        assert tsdata.shards.num_partitions() == 2

        tsdata = XShardsTSDataset.from_xshards(shards_multiple, dt_col="datetime",
                                               target_col=["value"],
                                               extra_feature_col="extra feature")
        assert tsdata._id_list == ['0']
        assert tsdata.feature_col == ["extra feature"]
        assert tsdata.target_col == ["value"]
        assert tsdata.dt_col == "datetime"
        assert tsdata.shards.num_partitions() == 1
        # stop_orca_context()

    def test_xshardstsdataset_split(self):
        # from bigdl.orca import init_orca_context, stop_orca_context
        # init_orca_context(cores=4, memory="2g")
        shards_multiple = read_csv(os.path.join(self.resource_path, "multiple.csv"))
        # only train and test
        tsdata_train, tsdata_valid, tsdata_test =\
            XShardsTSDataset.from_xshards(shards_multiple, dt_col="datetime", target_col="value",
                                          extra_feature_col=["extra feature"], id_col="id",
                                          with_split=True, val_ratio=0, test_ratio=0.1)
        # standard split with all three sets
        tsdata_train, tsdata_valid, tsdata_test =\
            XShardsTSDataset.from_xshards(shards_multiple, dt_col="datetime", target_col="value",
                                          extra_feature_col=["extra feature"], id_col="id",
                                          with_split=True, val_ratio=0.1, test_ratio=0.1,
                                          largest_look_back=5, largest_horizon=2)

        tsdata_train.feature_col.append("new extra feature")
        assert len(tsdata_train.feature_col) == 2
        assert len(tsdata_valid.feature_col) == 1
        assert len(tsdata_test.feature_col) == 1

        tsdata_train.target_col[0] = "new value"
        assert tsdata_train.target_col[0] == "new value"
        assert tsdata_valid.target_col[0] != "new value"
        assert tsdata_test.target_col[0] != "new value"
        # stop_orca_context()

    def test_xshardstsdataset_roll_multiple_id(self):
        # from bigdl.orca import init_orca_context, stop_orca_context
        # init_orca_context(cores=4, memory="2g")
        shards_multiple = read_csv(os.path.join(self.resource_path, "multiple.csv"))
        horizon = random.randint(1, 10)
        lookback = random.randint(1, 20)

        tsdata = XShardsTSDataset.from_xshards(shards_multiple, dt_col="datetime",
                                               target_col="value",
                                               extra_feature_col=["extra feature"], id_col="id")

        with pytest.raises(RuntimeError):
            tsdata.to_xshards()

        # roll train
        tsdata.roll(lookback=lookback, horizon=horizon)
        shards_numpy = tsdata.to_xshards()
        collected_numpy = shards_numpy.collect()  # collect and valid
        x = np.concatenate([collected_numpy[i]['x'] for i in range(len(collected_numpy))], axis=0)
        y = np.concatenate([collected_numpy[i]['y'] for i in range(len(collected_numpy))], axis=0)
        assert x.shape == ((50-lookback-horizon+1)*2, lookback, 2)
        assert y.shape == ((50-lookback-horizon+1)*2, horizon, 1)

        tsdata.roll(lookback=lookback, horizon=horizon,
                    feature_col=["extra feature"], target_col="value")
        shards_numpy = tsdata.to_xshards()
        collected_numpy = shards_numpy.collect()  # collect and valid
        x = np.concatenate([collected_numpy[i]['x'] for i in range(len(collected_numpy))], axis=0)
        y = np.concatenate([collected_numpy[i]['y'] for i in range(len(collected_numpy))], axis=0)
        assert x.shape == ((50-lookback-horizon+1)*2, lookback, 2)
        assert y.shape == ((50-lookback-horizon+1)*2, horizon, 1)

        tsdata.roll(lookback=lookback, horizon=horizon,
                    feature_col=[], target_col="value")
        shards_numpy = tsdata.to_xshards()
        collected_numpy = shards_numpy.collect()  # collect and valid
        x = np.concatenate([collected_numpy[i]['x'] for i in range(len(collected_numpy))], axis=0)
        y = np.concatenate([collected_numpy[i]['y'] for i in range(len(collected_numpy))], axis=0)
        assert x.shape == ((50-lookback-horizon+1)*2, lookback, 1)
        assert y.shape == ((50-lookback-horizon+1)*2, horizon, 1)

        # roll test
        horizon = 0
        lookback = random.randint(1, 20)

        tsdata.roll(lookback=lookback, horizon=horizon)
        shards_numpy = tsdata.to_xshards()
        collected_numpy = shards_numpy.collect()  # collect and valid
        x = np.concatenate([collected_numpy[i]['x'] for i in range(len(collected_numpy))], axis=0)
        assert x.shape == ((50-lookback-horizon+1)*2, lookback, 2)
        # stop_orca_context()
