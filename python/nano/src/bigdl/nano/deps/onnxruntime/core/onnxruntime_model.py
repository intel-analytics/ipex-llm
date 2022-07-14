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
from bigdl.nano.utils.log4Error import invalidInputError


class ONNXRuntimeModel:
    def __init__(self, onnx_filepath, session_options=None):
        self.onnx_filepath = onnx_filepath  # onnx filepath
        self.onnx_model = None  # onnx model
        self.ortsess = None  # onnxruntime session
        self._build_ortsess(session_options)

    def forward_step(self, *inputs):
        '''
        This function run through the onnxruntime forwarding step
        '''
        invalidInputError(len(self._forward_args) >= len(inputs), "The length of inputs is "
                          "inconsistent with the length of ONNX Runtime session's inputs, "
                          "there may be some redundant inputs.")
        inputs = dict(zip(self._forward_args, inputs))
        ort_outs = self.ortsess.run(None, inputs)
        return ort_outs

    def _build_ortsess(self,
                       sess_options=None):
        '''
        Internal function to build a ortsess.

        :param sess_options: ortsess options in ort.SessionOptions type
        '''
        self.onnx_model = onnx.load(self.onnx_filepath)
        self.ortsess = ort.InferenceSession(self.onnx_filepath, sess_options=sess_options)
        self._forward_args = list(map(lambda x: x.name, self.ortsess.get_inputs()))

    def _save_model(self, path):
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
