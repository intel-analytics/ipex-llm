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

import pytest

from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.common.nncontext import *
from zoo.chronos.preprocessing.impute import TimeMergeImputor


class TestImpute(ZooTestCase):
    def setup_method(self, method):
        sparkConf = init_spark_conf().setMaster("local[1]").setAppName("TestImpute")
        self.sc = init_nncontext(sparkConf)
        self.sqlContext = SQLContext(self.sc)

    def teardown_method(self, method):
        self.sc.stop()

    def test_time_merge_imputor(self):
        dict = [{'str_timestamp': "2020-11-09T07:52:00.000Z", 'value': 1},
                {'str_timestamp': "2020-11-09T07:52:03.000Z", 'value': 2},
                {'str_timestamp': "2020-11-09T07:52:09.000Z", 'value': 3}]
        df = self.sqlContext.createDataFrame(dict)
        from pyspark.sql.functions import to_timestamp
        df = df.withColumn("timestamp", to_timestamp(df['str_timestamp']))
        df.show(20, False)
        imputor = TimeMergeImputor(5, "timestamp", "max")
        imputor.impute(df).show(20, False)
        imputor = TimeMergeImputor(5, "timestamp")
        imputor.impute(df).show(20, False)
        imputor = TimeMergeImputor(5, "timestamp", "min")
        imputor.impute(df).show(20, False)
        imputor = TimeMergeImputor(5, "timestamp", "mean")
        imputor.impute(df).show(20, False)
        imputor = TimeMergeImputor(5, "timestamp", "sum")
        imputor.impute(df).show(20, False)

if __name__ == "__main__":
    pytest.main([__file__])
