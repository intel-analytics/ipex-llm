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
from bigdl.chronos.utils import LazyImport
XShardsTSDataset = LazyImport('bigdl.chronos.data.experimental.XShardsTSDataset')
read_csv = LazyImport('bigdl.orca.data.pandas.read_csv')
init_orca_context = LazyImport('bigdl.orca.common.init_orca_context')
stop_orca_context = LazyImport('bigdl.orca.common.stop_orca_context')
OrcaContext = LazyImport('bigdl.orca.common.OrcaContext')
from ... import op_torch, op_distributed

from pandas.testing import assert_frame_equal
from numpy.testing import assert_array_almost_equal


def generate_spark_df():
    sc = OrcaContext.get_spark_context()
    from pyspark.sql import SparkSession
    spark = SparkSession(sc)
    rdd = sc.range(0, 100)
    from pyspark.ml.linalg import DenseVector
    df = rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float)),
                            int(np.random.randint(0, 2, size=())),
                            int(x))).toDF(["feature", "id", "date"])
    return df

def get_ugly_ts_df():
    data = np.random.random_sample((100, 5))
    mask = np.random.random_sample((100, 5))
    newmask = mask.copy()
    mask[newmask >= 0.4] = 2
    mask[newmask < 0.4] = 1
    mask[newmask < 0.2] = 0
    data[mask == 0] = None
    data[mask == 1] = np.nan
    df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e'])
    df['a'][0] = np.nan  # make sure column 'a' has a N/A
    df["datetime"] = pd.date_range('1/1/2019', periods=100)
    df.loc[50:100, "datetime"] = pd.date_range('1/1/2019', periods=50)
    df["id"] = np.array(['00']*50 + ['01']*50)
    return df

@op_torch
@op_distributed
class TestXShardsTSDataset(TestCase):

    def setUp(self):
        # In read_csv module, the initialized sc cannot be used normally
        # Here, force initialize
        init_orca_context(cores=8)
        self.resource_path = os.path.join(os.path.split(__file__)[0], "../../resources/")

    def tearDown(self):
        pass
    
    @classmethod
    def tearDownClass(cls):
        # stop possible active_spark_context
        from pyspark import SparkContext
        from bigdl.orca.ray import OrcaRayContext
        if SparkContext._active_spark_context is not None:
            print("Stopping spark_orca context")
            sc = SparkContext.getOrCreate()
            if sc.getConf().get("spark.master").startswith("spark://"):
                from bigdl.dllib.nncontext import stop_spark_standalone
                stop_spark_standalone()
            sc.stop()

    def test_xshardstsdataset_initialization(self):
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

    def test_xshardstsdataset_initialization_multiple(self):
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

    def test_xshardstsdataset_split(self):
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
                                          with_split=True, val_ratio=0.1, test_ratio=0.1)

        tsdata_train.feature_col.append("new extra feature")
        assert len(tsdata_train.feature_col) == 2
        assert len(tsdata_valid.feature_col) == 1
        assert len(tsdata_test.feature_col) == 1

        tsdata_train.target_col[0] = "new value"
        assert tsdata_train.target_col[0] == "new value"
        assert tsdata_valid.target_col[0] != "new value"
        assert tsdata_test.target_col[0] != "new value"

    def test_xshardstsdataset_roll_multiple_id(self):
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

    def test_xshardstsdataset_scale_unscale(self):
        from sklearn.preprocessing import StandardScaler
        scalers = [{0: StandardScaler(), 1: StandardScaler()}]
        df = pd.read_csv(os.path.join(self.resource_path, "multiple.csv"))
        for scaler in scalers:
            shards_multiple = read_csv(os.path.join(self.resource_path, "multiple.csv"), dtype={"id": np.int64})
            shards_multiple_test = read_csv(os.path.join(self.resource_path, "multiple.csv"), dtype={"id": np.int64})

            tsdata = XShardsTSDataset.from_xshards(shards_multiple, dt_col="datetime",
                                                   target_col="value",
                                                   extra_feature_col=["extra feature"], id_col="id")
            tsdata_test = XShardsTSDataset.from_xshards(shards_multiple_test, dt_col="datetime",
                                                        target_col="value",
                                                        extra_feature_col=["extra feature"], id_col="id")

            tsdata.scale(scaler)
            tsdata_test.scale(scaler, fit=False)

            def get_local_df(tsdata):
                collected_df = tsdata.shards.collect()
                collected_df = pd.concat(collected_df, axis=0)
                collected_df.reset_index(inplace=True)
                collected_df["datetime"] = df["datetime"]
                del collected_df["index"]
                return collected_df

            df_train = get_local_df(tsdata)
            df_test = get_local_df(tsdata_test)
            assert_frame_equal(df_train, df_test)

            with pytest.raises(AssertionError):
                assert_frame_equal(df_train, df)
                assert_frame_equal(df_test, df)

            tsdata.unscale()
            df_train_unscale = get_local_df(tsdata)
            assert_frame_equal(df_train_unscale, df)
            
    def test_xshardstsdataset_unscale_numpy(self):
        from sklearn.preprocessing import  StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler
        scalers = [{0: StandardScaler(), 1: StandardScaler()}, 
                   {0: MaxAbsScaler(), 1: MaxAbsScaler()},
                   {0: MinMaxScaler(), 1: MinMaxScaler()},
                   {0: RobustScaler(), 1: RobustScaler()}]
        for scaler in scalers:
            shards_multiple = read_csv(os.path.join(self.resource_path, "multiple.csv"))

            tsdata = XShardsTSDataset.from_xshards(shards_multiple, dt_col="datetime",
                                                   target_col="value",
                                                   extra_feature_col=["extra feature"], id_col="id")

            ori_tsdata = XShardsTSDataset.from_xshards(shards_multiple, dt_col="datetime",
                                                   target_col="value",
                                                   extra_feature_col=["extra feature"], id_col="id")
            horizon = random.randint(1, 10)
            lookback = random.randint(1, 20)
            ori_tsdata.roll(horizon=horizon, lookback=lookback)
            ori_arr_shards = ori_tsdata.to_xshards().collect()
            ori_arr_shards = [arr['y'] for arr in ori_arr_shards]
            ori_y = np.concatenate(ori_arr_shards, axis=0)
            tsdata.scale(scaler)
            tsdata.roll(horizon=horizon, lookback=lookback)

            scale_arr = tsdata.to_xshards()
            numpy_tsdata = tsdata.unscale_xshards(scale_arr, 'y')
            numpy_tsdata = numpy_tsdata.collect()
            y = np.concatenate(numpy_tsdata, axis=0)
            assert_array_almost_equal(ori_y, y)

    def test_xshardstsdataset_impute(self):
        from tempfile import TemporaryDirectory
        tmp_df = get_ugly_ts_df()
        with TemporaryDirectory() as tmpdir:
            file_name = os.path.join(tmpdir, 'impute.csv')
            tmp_df.to_csv(file_name, index=False)
            shards_tmp = read_csv(file_name)
        
            for val in ["last", "const", "linear"]:
                tsdata = XShardsTSDataset.from_xshards(shards_tmp, 
                                                    dt_col="datetime", target_col="e",
                                            extra_feature_col=["a", "b", "c", "d"], id_col="id")
                tsdata.impute(mode=val)
                collected_df = tsdata.shards.collect()
                collected_df = pd.concat(collected_df, axis=0)
                
                assert collected_df.isna().sum().sum() == 0
                assert len(collected_df) == 100

    def test_xshardstsdataset_sparkdf(self):
        df = generate_spark_df()

        # with id
        tsdata = XShardsTSDataset.from_sparkdf(df, dt_col="date",
                                               target_col="feature",
                                               id_col="id")
        tsdata.roll(lookback=4, horizon=2)
        data = tsdata.to_xshards(partition_num=5)
        data_collected = data.collect()
        assert data_collected[0]['x'].shape[1] == 4
        assert data_collected[0]['x'].shape[2] == 1
        assert data_collected[0]['y'].shape[1] == 2
        assert data_collected[0]['y'].shape[2] == 1
        assert data.num_partitions() == 5
        assert "id" in data.collect()[0].keys()
        assert tsdata.shards.num_partitions() == 2

        # with only 1 id
        tsdata = XShardsTSDataset.from_sparkdf(df, dt_col="date",
                                               target_col="feature")
        tsdata.roll(lookback=4, horizon=2)
        data = tsdata.to_xshards().collect()
        assert data[0]['x'].shape[1] == 4
        assert data[0]['x'].shape[2] == 1
        assert data[0]['y'].shape[1] == 2
        assert data[0]['y'].shape[2] == 1
        assert tsdata.shards.num_partitions() == 1
