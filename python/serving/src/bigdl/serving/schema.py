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

import pyarrow as pa
import numpy as np
import cv2
import base64
from bigdl.serving.log4Error import invalidInputError


def get_field_and_data(key, value):
    if isinstance(value, list):
        invalidInputError(len(value) > 0, "empty list is not supported")
        sample = value[0]
        if isinstance(sample, str):
            # list of string will be converted to Tensor of String
            # use | to split
            str_concat = '|'.join(value)
            field = pa.field(key, pa.string())
            data = pa.array([str_concat])
            return field, data

        elif isinstance(sample, np.ndarray):
            invalidInputError(len(value) == 3,
                              "Sparse Tensor must have list of ndarray with length 3, which"
                              " represent indices, values, shape respectively")
            indices_field = pa.field("indiceData", pa.list_(pa.int32()))
            indices_shape_field = pa.field("indiceShape", pa.list_(pa.int32()))
            value_field = pa.field("data", pa.list_(pa.float32()))
            shape_field = pa.field("shape", pa.list_(pa.int32()))
            sparse_tensor_type = pa.struct(
                [indices_field, indices_shape_field, value_field, shape_field])
            field = pa.field(key, sparse_tensor_type)

            shape = value[2]
            values = value[1]
            indices = value[0].astype("float32").flatten()
            indices_shape = value[0].shape
            data = pa.array([{'indiceData': indices},
                             {'indiceShape': indices_shape},
                             {'data': values},
                             {'shape': shape}], type=sparse_tensor_type)
            return field, data
        else:
            invalidInputError(False,
                              "List of string and ndarray is supported,"
                              "but your input does not match")

    elif isinstance(value, str):
        # str value will be considered as image path
        field = pa.field(key, pa.string())
        data = encode_image(value)
        # b = bytes(data, "utf-8")
        data = pa.array([data])
        # ba = pa.array(b, type=pa.binary())
        return field, data
    elif isinstance(value, dict):
        if "path" in value.keys():
            path = value["path"]
            data = encode_image(path)
        elif "b64" in value.keys():
            data = value["b64"]
        else:
            invalidInputError(False,
                              "Your input dict must contain"
                              " either 'path' or 'b64' key")
        field = pa.field(key, pa.string())
        data = pa.array([data])
        return field, data

    elif isinstance(value, np.ndarray):
        # ndarray value will be considered as tensor
        indices_field = pa.field("indiceData", pa.list_(pa.int32()))
        indices_shape_field = pa.field("indiceShape", pa.list_(pa.int32()))
        data_field = pa.field("data", pa.list_(pa.float32()))
        shape_field = pa.field("shape", pa.list_(pa.int32()))
        tensor_type = pa.struct(
            [indices_field, indices_shape_field, data_field, shape_field])
        field = pa.field(key, tensor_type)

        shape = np.array(value.shape)
        d = value.astype("float32").flatten()
        # data = pa.array([{'data': d}, {'shape': shape}, {}],
        #                 type=tensor_type)
        data = pa.array([{'indiceData': []},
                         {'indiceShape': []},
                         {'data': d},
                         {'shape': shape}], type=tensor_type)
        return field, data

    else:
        invalidInputError(False,
                          "Your request does not match any schema, "
                          "please check.")


def encode_image(img):
    """
    :param id: String you use to identify this record
    :param data: Data, ndarray type
    :return:
    """
    if isinstance(img, str):
        img = cv2.imread(img)
        if img.size == 0:
            print("You have pushed an image with path: ",
                  img, "the path is invalid, skipped.")
            return

    # force resize here to avoid input image shape inconsistent
    # if the shape is consistent, it would not affect the data
    data = cv2.imencode(".jpg", img)[1]
    img_encoded = base64.b64encode(data).decode("utf-8")
    return img_encoded
