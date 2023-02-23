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
import ray
import pytest
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.sql import SparkSession


@pytest.fixture(autouse=True, scope='package')
def orca_context_fixture():
    from bigdl.orca import init_orca_context, stop_orca_context
    sc = init_orca_context(cores=8)

    def to_array_(v):
        return v.toArray().tolist()

    def flatten_(v):
        result = []
        for elem in v:
            result.extend(elem.toArray().tolist())
        return result

    spark = SparkSession(sc)
    spark.udf.register("to_array", to_array_, ArrayType(DoubleType()))
    spark.udf.register("flatten", flatten_, ArrayType(DoubleType()))
    yield
    stop_orca_context()
