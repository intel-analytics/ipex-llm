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
from bigdl.orca.data import SparkXShards

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from bigdl.orca.data.tf.data import Dataset
    from bigdl.orca.data.shard import SparkXShards
    from bigdl.orca.data.ray_xshards import RayXShards


class TF2Dataset(object):
    def __init__(self, dataset: "Dataset") -> None:
        self.rdd = dataset.as_tf_dataset_rdd()
        self.dataset = dataset

    def get_origin_xshards(self) -> "SparkXShards":
        return self.dataset.get_xshards()

    def get_xshards(self) -> "SparkXShards":
        return SparkXShards(self.rdd)

    def get_ray_xshards(self, num_workers: int) -> "RayXShards":
        from bigdl.orca.data.utils import process_spark_xshards
        xshards = self.get_xshards()
        return process_spark_xshards(xshards, num_workers)
