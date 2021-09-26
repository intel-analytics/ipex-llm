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
import pytest
import numpy as np
from bigdl.orca.data import XShards


class TestSparkBackend(TestCase):

    def test_partition_ndarray(self):

        data = np.random.randn(10, 4)

        xshards = XShards.partition(data)

        data_parts = xshards.rdd.collect()

        reconstructed = np.concatenate(data_parts)
        assert np.allclose(data, reconstructed)

    def test_partition_tuple(self):
        data1 = np.random.randn(10, 4)
        data2 = np.random.randn(10, 4)

        xshards = XShards.partition((data1, data2))

        data_parts = xshards.rdd.collect()

        data1_parts = [part[0] for part in data_parts]
        data2_parts = [part[1] for part in data_parts]

        reconstructed1 = np.concatenate(data1_parts)
        reconstructed2 = np.concatenate(data2_parts)
        assert np.allclose(data1, reconstructed1)
        assert np.allclose(data2, reconstructed2)

    def test_partition_list(self):
        data1 = np.random.randn(10, 4)
        data2 = np.random.randn(10, 4)

        xshards = XShards.partition([data1, data2])

        data_parts = xshards.rdd.collect()

        data1_parts = [part[0] for part in data_parts]
        data2_parts = [part[1] for part in data_parts]

        reconstructed1 = np.concatenate(data1_parts)
        reconstructed2 = np.concatenate(data2_parts)
        assert np.allclose(data1, reconstructed1)
        assert np.allclose(data2, reconstructed2)

    def test_partition_dict(self):
        data1 = np.random.randn(10, 4)
        data2 = np.random.randn(10, 4)

        xshards = XShards.partition({"x": data1, "y": data2})

        data_parts = xshards.rdd.collect()

        data1_parts = [part["x"] for part in data_parts]
        data2_parts = [part["y"] for part in data_parts]

        reconstructed1 = np.concatenate(data1_parts)
        reconstructed2 = np.concatenate(data2_parts)
        assert np.allclose(data1, reconstructed1)
        assert np.allclose(data2, reconstructed2)

    def test_partition_nested(self):
        data1 = np.random.randn(10, 4)
        data2 = np.random.randn(10, 4)

        xshards = XShards.partition({"x": (data1, ), "y": [data2]})

        data_parts = xshards.rdd.collect()

        data1_parts = [part["x"][0] for part in data_parts]
        data2_parts = [part["y"][0] for part in data_parts]

        reconstructed1 = np.concatenate(data1_parts)
        reconstructed2 = np.concatenate(data2_parts)
        assert np.allclose(data1, reconstructed1)
        assert np.allclose(data2, reconstructed2)

    def test_partition_ndarray_with_num_shards_specification(self):
        data = np.random.randn(10, 4)
        # Reasonable number of shards
        xshards = XShards.partition(data, num_shards=2)

        data_parts = xshards.rdd.collect()

        reconstructed = np.concatenate(data_parts)
        assert np.allclose(data, reconstructed)
        # Empty shards
        with pytest.raises(ValueError) as errorInfo:
            xshards = XShards.partition(data, num_shards=20)

        assert errorInfo.type == ValueError
        assert "number of shards" in str(errorInfo.value)

    def test_partition_nested_with_num_shards_specification(self):
        data1 = np.random.randn(10, 4)
        data2 = np.random.randn(10, 4)
        # Reasonable number of shards
        xshards = XShards.partition({"x": (data1, ), "y": [data2]}, num_shards=2)

        data_parts = xshards.rdd.collect()

        data1_parts = [part["x"][0] for part in data_parts]
        data2_parts = [part["y"][0] for part in data_parts]

        reconstructed1 = np.concatenate(data1_parts)
        reconstructed2 = np.concatenate(data2_parts)
        assert np.allclose(data1, reconstructed1)
        assert np.allclose(data2, reconstructed2)
        # Empty shards
        with pytest.raises(ValueError) as errorInfo:
            xshards = XShards.partition({"x": (data1, ), "y": [data2]}, num_shards=20)

        assert errorInfo.type == ValueError
        assert "number of shards" in str(errorInfo.value)


if __name__ == "__main__":
    pytest.main([__file__])
