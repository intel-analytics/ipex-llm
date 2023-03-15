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

from bigdl.orca.data import spark_df_to_ray_dataset

from bigdl.orca import OrcaContext
from pyspark.sql import SparkSession


def test_spark_df_to_ray_dataset(orca_context_fixture):
    sc = OrcaContext.get_spark_context()
    spark = SparkSession(sc)
    spark_df = spark.createDataFrame([(1, "a"), (2, "b"), (3, "c"), (4, "d")], ["one", "two"])
    rows = [(r.one, r.two) for r in spark_df.take(4)]
    ds = spark_df_to_ray_dataset(spark_df)
    values = [(r["one"], r["two"]) for r in ds.take(8)]
    assert values == rows

if __name__ == "__main__":
    pytest.main([__file__])
