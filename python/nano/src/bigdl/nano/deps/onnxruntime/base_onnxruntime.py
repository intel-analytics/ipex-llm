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
from bigdl.nano.utils.log4Error import invalidInputError

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    invalidInputError(False,
                      "To enable onnxruntime inference, you need to install it by:\n"
                      "\t\t pip install onnxruntime")


def save_onnx_to_file(onnx_model, file_path="model.onnx"):
    onnx.save(onnx_model, file_path)


class BaseORTInference:
    def __init__(self):
        self.ortsess_fp32 = None  # onnxruntime session fp32
        self.onnx_model_fp32 = None  # onnx model fp32
        self.onnx_filepath_fp32 = None  # onnx filepath fp32
        self.ortsess_int8 = None  # onnxruntime session int8
        self.onnx_model_int8 = None  # onnx model int8
        self.onnx_filepath_int8 = None  # onnx filepath int8
        self.example_input_array = None  # cached example input array

    def forward_step(self, *inputs):
        '''
            This function run through the onnxruntime forwarding step
            with presision='fp32'
        '''
        inputs = dict(zip(self._forward_args, inputs))
        ort_outs = self.ortsess_fp32.run(None, inputs)
        return ort_outs

    def forward_step_int8(self, *inputs):
        '''
            This function run through the onnxruntime forwarding step
            with presision='fp32'
        '''
        inputs = dict(zip(self._forward_args, inputs))
        ort_outs = self.ortsess_int8.run(None, inputs)
        return ort_outs

    def _build_ortsess_fp32(self,
                            sess_options=None):
        '''
        Internal function to build a ortsess.

        :param sess_options: ortsess options in ort.SessionOptions type
        '''
        self.onnx_model_fp32 = onnx.load(self.onnx_filepath_fp32)
        self.ortsess_fp32 = ort.InferenceSession(self.onnx_filepath_fp32, sess_options=sess_options)

    def _build_ortsess_int8(self,
                            sess_options=None):
        '''
        Internal function to build a ortsess.

        :param sess_options: ortsess options in ort.SessionOptions type
        '''
        self.onnx_model_int8 = onnx.load(self.onnx_filepath_int8)
        self.ortsess_int8 = ort.InferenceSession(self.onnx_filepath_int8, sess_options=sess_options)

    def reset(self, model):
        self.ortsess_fp32 = None  # onnxruntime session fp32
        self.onnx_model_fp32 = None  # onnx model fp32
        self.onnx_filepath_fp32 = None  # onnx filepath fp32
        self.ortsess_int8 = None  # onnxruntime session int8
        self.onnx_model_int8 = None  # onnx model int8
        self.onnx_filepath_int8 = None  # onnx filepath int8
