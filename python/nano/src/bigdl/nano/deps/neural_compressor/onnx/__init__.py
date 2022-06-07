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
from ..core import version
from packaging import version as v
from bigdl.nano.utils.log4Error import invalidInputError

from .pytorch.quantization import PytorchONNXRuntimeQuantization

if v.parse(version) >= v.parse("1.11"):
    try:
        import onnxruntime_extensions
    except ImportError:
        invalidInputError(
            False,
            errMsg="Neural Compressor >=1.11 requires onnxruntime_extensions.",
            fixMsg="Please run installation:\n\t pip install onnxruntime-extensions"
        )
