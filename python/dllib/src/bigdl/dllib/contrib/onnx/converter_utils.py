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

import math
import numpy as np
from bigdl.dllib.utils.log4Error import *


def calc_output_shape(input, kernel, padding=0, stride=1, dilation=1, ceil_mode=False):
    def dilated_kernel_size(kernel, dilation):
        return kernel + (kernel - 1) * (dilation - 1)

    rounding = math.ceil if ceil_mode else math.floor
    out = (input + 2 * padding - dilated_kernel_size(kernel, dilation)) / stride + 1
    out = int(rounding(out))
    return out


def parse_node_attr(node_proto):
    attrs = {}
    attr_proto = node_proto.attribute

    for attr in attr_proto:
        for field in ['f', 'i', 's']:
            if attr.HasField(field):
                attrs[attr.name] = getattr(attr, field)

                # Needed for supporting python version > 3.5
                if isinstance(attrs[attr.name], bytes):
                    attrs[attr.name] = attrs[attr.name].decode(encoding='utf-8')

        for field in ['floats', 'ints', 'strings']:
            if list(getattr(attr, field)):
                invalidInputError(attr.name not in attrs,
                                  "Only one type of attr is allowed")
                attrs[attr.name] = tuple(getattr(attr, field))

        for field in ['t', 'g']:
            if attr.HasField(field):
                attrs[attr.name] = getattr(attr, field)
        for field in ['tensors', 'graphs']:
            if list(getattr(attr, field)):
                invalidInputError(False, "Not implement yet")
        if attr.name not in attrs:
            invalidInputError(False, "Cannot parse attribute: \n{}\n.".format(attr))

    return attrs


def parse_tensor_data(tensor_proto):
    try:
        from onnx.numpy_helper import to_array
    except ImportError:
        invalidInputError(False, "Onnx and protobuf need to be installed.")
    if len(tuple(tensor_proto.dims)) > 0:
        np_array = to_array(tensor_proto).reshape(tuple(tensor_proto.dims))
    else:
        # If it is a scalar tensor
        np_array = np.array([to_array(tensor_proto)])
    return np_array
