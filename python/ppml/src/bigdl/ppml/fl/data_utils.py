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

import pandas as pd
import numpy as np
from bigdl.ppml.fl import *
from bigdl.dllib.utils.log4Error import invalidInputError


def get_input_type(x, y=None):
    if isinstance(x, pd.DataFrame):
        if y is not None and not isinstance(y, pd.DataFrame):
            invalidInputError(False,
                              f"Feature is DataFrame, label should be DataFrame,"
                              f" but got {type(y)}")
        return "DataFrame"
    elif isinstance(x, np.ndarray):
        if y is not None and not isinstance(y, np.ndarray):
            invalidInputError(False,
                              f"Feature is Numpy NdArray, label should be Numpy NdArray,"
                              f" but got {type(y)}")
        return "NdArray"
    else:
        invalidInputError(False,
                          f"Supported argument types: DataFrame, NdArray, but got {type(x)}")


def convert_to_numpy(x, columns=None):
    if isinstance(x, pd.DataFrame):
        if columns is not None:
            return x[columns].to_numpy()
        else:
            return x.to_numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        invalidInputError(False,
                          f"{type(x)} can not be converted to numpy")


def convert_to_jtensor(x, y=None, feature_columns=None, label_columns=None):
    arg_type = get_input_type(x, y)
    if arg_type == "DataFrame":
        if feature_columns is None or (y is not None and label_columns is None):
            invalidInputError(False,
                              "Input DataFrame type must have feature_columns and label_columns")
        x = x[feature_columns].to_numpy()
        y = y[label_columns].to_numpy() if y is not None else None
    return JTensor.from_ndarray(x), JTensor.from_ndarray(y)




