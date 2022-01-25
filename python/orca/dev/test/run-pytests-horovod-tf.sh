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

set -ex

cd "`dirname $0`"

export PYSPARK_PYTHON=python
export PYSPARK_DRIVER_PYTHON=python

ray stop -f

cd ../../
echo "Running RayOnSpark tests"
python -m pytest -v test/bigdl/orca/learn/ray/tf/
    --ignore=test/bigdl/orca/learn/ray/tf/test_ray_tf2estimator.py
python -m pytest -v test/bigdl/orca/data/test_read_parquet_images.py
