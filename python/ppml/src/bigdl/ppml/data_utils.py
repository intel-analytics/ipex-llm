#
# Copyright 2021 The BigDL Authors.
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


def convert_to_numpy(x, dataframe_columns=None):
    """
    :param x: The input to convert
    :param dataframe_columns: applicable if x is pandas.DataFrame, the column to convert
    :return: the converted numpy.ndarray
    """
    if isinstance(x, pd.DataFrame):
        return [x[col] for col in dataframe_columns]
    elif isinstance(x, np.ndarray):
        return [x]
    elif isinstance(x, list):
        for e in x:
            if not isinstance(x, np.ndarray):
                raise Exception("only NdArray type is supported for list input")
        return x
    else:
        raise Exception("Input could be Pandas DataFrame or Numpy NDArray or list of NDArray, but got", type(x))


