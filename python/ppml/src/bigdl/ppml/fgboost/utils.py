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

import numpy as np
from sys import getsizeof
import math
from bigdl.dllib.utils.common import *

# 2 ** 32 is the JVM INT limit, we reserve 2 ^ 4 here to make it safer
# e.g. some size compute step may multiply to cause bound exceed

MAX_MSG_SIZE = 2 ** 28 
def add_data(data: np.ndarray, jvalue, func_add, bigdl_type="float"):
    size = getsizeof(data)
    batch_num = math.ceil(size / MAX_MSG_SIZE)
    data_per_batch = data.shape[0] / batch_num
    for i in range(batch_num):
        idx = i * data_per_batch
        data_batch = data[idx:idx + data_per_batch, ...]
        callBigDlFunc(bigdl_type, func_add, jvalue, data)

