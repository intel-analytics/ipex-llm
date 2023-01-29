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
from onnxsim import simplify
import onnx


def onnx_simplify(onnx_path: str):
    """
    Simplify the ONNX model based on onnxsim.
    If simplification is successful, will overwrite new ONNX model to onnx_path

    :param onnx_path: File path of of onnx ModelProto object.
    """
    # load your predefined ONNX model
    model = onnx.load(onnx_path)

    # convert model
    model_simp, check = simplify(model)

    # overwrite model_simp to onnx_path
    if check is True:
        onnx.save(model_simp, onnx_path)
