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


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def index(x, start=None, end=None):
    if isinstance(x, list):
        return [index_numpy(x_i, start, end) for x_i in x]
    else:
        return index_numpy(x, start, end)


def index_numpy(x, start=None, end=None):
    if start:
        if end:
            return x[start:end]
        else:
            return x[start:]
    else:
        if end:
            return x[:end]
        else:
            return x


def process_records(buffer):
    import random
    random.shuffle(buffer)  # TODO: Make shuffle configurable?
    buffer_x = [record[0] for record in buffer]
    if len(buffer[0]) > 1:
        buffer_y = [record[1] for record in buffer]
    else:
        buffer_y = None
    res_buffer = dict()
    if isinstance(buffer_x[0], list):
        res_x = []
        for i in range(len(buffer_x[0])):
            res_x.append(cast_ndarray_type(np.array([record[i] for record in buffer_x])))
        res_buffer["x"] = res_x
    else:
        res_buffer["x"] = cast_ndarray_type(np.array(buffer_x))
    if buffer_y:
        if isinstance(buffer_y[0], list):
            res_y = []
            for i in range(len(buffer_x[0])):
                res_y.append(cast_ndarray_type(np.array([record[i] for record in buffer_y])))
            res_buffer["y"] = res_y
        else:
            res_buffer["y"] = cast_ndarray_type(np.array(buffer_y))
    return res_buffer


# To save some memory when putting into plasma
def cast_ndarray_type(x):
    if x.dtype == np.int64:
        return x.astype(np.int32)
    elif x.dtype == np.float64:
        return x.astype(np.float32)
    else:
        return x
