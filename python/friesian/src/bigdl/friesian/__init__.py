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

from bigdl.dllib.nncontext import *
from bigdl.dllib.utils.zoo_engine import prepare_env, is_spark_below_ver

prepare_env()
if not is_spark_below_ver("2.4"):
    JavaCreator.add_creator_class("com.intel.analytics.bigdl.friesian.python.PythonFriesian")
else:
    warnings.warn("You are strongly recommended to use Spark > 2.4 for Friesian")
