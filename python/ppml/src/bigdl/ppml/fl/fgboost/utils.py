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

import time
import numpy as np
from sys import getsizeof
import math
from bigdl.dllib.utils.common import *
from bigdl.ppml.fl.data_utils import convert_to_jtensor

# 2 ** 32 is the JVM INT limit, we reserve 2 ^ 4 here to make it safer
# e.g. some size compute step may multiply to cause bound exceed

MAX_MSG_SIZE = 2 ** 28 
def add_data(data: np.ndarray, jvalue, func_add, bigdl_type="float"):
    size = getsizeof(data)
    batch_num = math.ceil(size / MAX_MSG_SIZE)
    data_per_batch = math.ceil(data.shape[0] / batch_num)
    for i in range(batch_num):
        idx = i * data_per_batch
        data_batch = data[idx:idx + data_per_batch, ...] if i != batch_num - 1 else data[idx:, ...]
        ts = time.time()
        data_batch, _ = convert_to_jtensor(data_batch)
        te_convert = time.time()
        logging.info(f"numpy to jtensor time: {te_convert - ts}")
        callBigDlFunc(bigdl_type, func_add, jvalue, data_batch)
        te_add = time.time()
        logging.info(f"call add data time: {te_add - te_convert}")

