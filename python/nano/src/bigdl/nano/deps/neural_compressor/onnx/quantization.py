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
from bigdl.nano.deps.onnxruntime.core.onnxruntime_model import ONNXRuntimeModel
from ..core import BaseQuantization


class BaseONNXRuntimeQuantization(BaseQuantization):
    def __init__(self, framework='onnxrt_qlinear', **kwargs):
        """
        Create a Intel Neural Compressor Quantization object for ONNXRuntime.
        """
        kwargs['framework'] = framework
        super().__init__(**kwargs)

    @property
    def valid_frameworks(self):
        return ('onnxrt_qlinearops', 'onnxrt_integerops')
