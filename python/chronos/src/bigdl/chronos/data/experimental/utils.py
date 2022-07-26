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


def add_row(df, name, const_num):
    df[name] = const_num
    return df


def transform_to_dict(data):
    if data[1] is None:
        return {"x": data[0].astype(np.float32),
                "id": data[2].astype(np.float32)}
    return {"x": data[0].astype(np.float32),
            "y": data[1].astype(np.float32),
            "id": data[2].astype(np.float32)}
