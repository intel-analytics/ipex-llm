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
from bigdl.util.common import Sample as BSample, JTensor as BJTensor, callBigDlFunc
import numpy as np


def to_list_of_numpy(elements):

    if isinstance(elements, np.ndarray):
        return [elements]
    elif np.isscalar(elements):
        return [np.array(elements)]
    elif not isinstance(elements, list):
        raise ValueError("Wrong type: %s" % type(elements))

    results = []
    for element in elements:
        if np.isscalar(element):
            results.append(np.array(element))
        elif isinstance(element, np.ndarray):
            results.append(element)
        else:
            raise ValueError("Wrong type: %s" % type(element))

    return results


def set_core_number(num):
    callBigDlFunc("float", "setCoreNumber", num)


class JTensor(BJTensor):

    def __init__(self, storage, shape, bigdl_type="float", indices=None):
        super(JTensor, self).__init__(storage, shape, bigdl_type, indices)

    @classmethod
    def from_ndarray(cls, a_ndarray, bigdl_type="float"):
        """
        Convert a ndarray to a DenseTensor which would be used in Java side.
        """
        if a_ndarray is None:
            return None
        assert isinstance(a_ndarray, np.ndarray), \
            "input should be a np.ndarray, not %s" % type(a_ndarray)
        return cls(a_ndarray,
                   a_ndarray.shape,
                   bigdl_type)


class Sample(BSample):

    def __init__(self, features, labels, bigdl_type="float"):
        super(Sample, self).__init__(features, labels, bigdl_type)

    @classmethod
    def from_ndarray(cls, features, labels, bigdl_type="float"):
        features = to_list_of_numpy(features)
        labels = to_list_of_numpy(labels)
        return cls(
            features=[JTensor(feature, feature.shape) for feature in features],
            labels=[JTensor(label, label.shape) for label in labels],
            bigdl_type=bigdl_type)
