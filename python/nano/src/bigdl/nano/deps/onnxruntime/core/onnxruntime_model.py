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
from pathlib import Path
import onnxruntime as ort
import onnx
from bigdl.nano.utils.common import invalidInputError, _flatten
import numpy as np


class ONNXRuntimeModel:
    def __init__(self, onnx_filepath, session_options=None):
        self.onnx_filepath = onnx_filepath  # onnx filepath
        self.onnx_model = None  # onnx model
        self.ortsess = None  # onnxruntime session
        self.session_options = session_options
        self._build_ortsess(session_options)

    def forward_step(self, *inputs, **kwargs):
        '''
        This function run through the onnxruntime forwarding step
        '''
        flattened_inputs = []
        _flatten(inputs, flattened_inputs)
        zipped_inputs = dict(zip(self.forward_args, flattened_inputs))
        if kwargs is not None and len(kwargs) > 0:
            zipped_inputs.update(kwargs)
        # TODO: kwargs has tuple or dict?
        if len(self._forward_args) != len(zipped_inputs):
            # formatting a Tensor will cost much time,
            # so we put it in this `if` statement
            invalidInputError(False,
                              "The length of inputs is "
                              "inconsistent with the length of ONNX Runtime session's inputs, "
                              f"got model_forward_args: {self._forward_args}, "
                              f"and flattened inputs: {flattened_inputs}")
        ort_outs = self.ortsess.run(None, zipped_inputs)
        return ort_outs

    @property
    def forward_args(self):
        return self._forward_args

    def _build_ortsess(self,
                       sess_options=None):
        '''
        Internal function to build a ortsess.

        :param sess_options: ortsess options in ort.SessionOptions type
        '''
        onnx_path_or_bytes = self.onnx_filepath
        if isinstance(self.onnx_filepath, str):
            self.onnx_model = onnx.load(self.onnx_filepath)
        elif isinstance(self.onnx_filepath, bytes):
            self.onnx_model = onnx.load_model_from_string(self.onnx_filepath)
        else:
            invalidInputError(
                isinstance(self.onnx_filepath, onnx.ModelProto),
                errMsg="Model type {} is not a legal ONNX model.".format(type(self.onnx_filepath))
            )
            self.onnx_model = self.onnx_filepath
            onnx_path_or_bytes = self.onnx_filepath.SerializeToString()
        self.ortsess = ort.InferenceSession(onnx_path_or_bytes, sess_options=sess_options)
        self._forward_args = list(map(lambda x: x.name, self.ortsess.get_inputs()))

    def _save_model(self, path, compression="fp32"):
        """
        Save ONNXRuntimeModel to local as an onnx file

        :param path: Path to save the model.
        """
        path = Path(path)
        invalidInputError(self.onnx_model,
                          "self.ie_network shouldn't be None.")
        invalidInputError(path.suffix == ".onnx",
                          "Path of onnx model must be with '.onnx' suffix.")
        onnx.save(self.onnx_model, str(path))
