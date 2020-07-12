#
# Copyright 2018 Analytics Zoo Authors.
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
from functools import reduce
import numpy as np


def get_data_label(partition_data):
    def combine_dict(dict1, dict2):
        return {key: np.concatenate((value, dict2[key]), axis=0)
                for (key, value) in dict1.items()}

    def combine_list(list1, list2):
        return [np.concatenate((list1[index], list2[index]), axis=0)
                for index in range(0, len(list1))]

    data_list = [data['x'] for data in partition_data]
    label_list = [data['y'] for data in partition_data]
    if isinstance(partition_data[0]['x'], dict):
        data = reduce(lambda dict1, dict2: combine_dict(dict1, dict2), data_list)
    elif isinstance(partition_data[0]['x'], np.ndarray):
        data = reduce(lambda array1, array2: np.concatenate((array1, array2), axis=0),
                      data_list)
    elif isinstance(partition_data[0]['x'], list):
        data = reduce(lambda list1, list2: combine_list(list1, list2), data_list)

    if isinstance(partition_data[0]['y'], dict):
        label = reduce(lambda dict1, dict2: combine_dict(dict1, dict2), label_list)
    elif isinstance(partition_data[0]['y'], np.ndarray):
        label = reduce(lambda array1, array2: np.concatenate((array1, array2), axis=0),
                       label_list)
    elif isinstance(partition_data[0]['y'], list):
        label = reduce(lambda list1, list2: combine_list(list1, list2), data_list)

    return data, label
