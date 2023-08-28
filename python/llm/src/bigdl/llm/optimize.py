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

from .transformers import ggml_convert_quant
from bigdl.llm.ggml.quantize import ggml_tensor_qtype
from bigdl.llm.utils.common import invalidInputError

def optimize_model(model, load_in_low_bit, optimize_atten=True):
    """
    A method to optimize any pytorch models.
    
    :param model: The original PyTorch model (nn.module)
    :param load_in_low_bit: Supported low-bit options are "sym_int4", "asym_int4", "sym_int5", 
        "asym_int5" or "sym_int8".
    :param optimize_atten: Whether to further optimize attention.
        
    return: The optimized model.
    """
    invalidInputError(load_in_low_bit in ggml_tensor_qtype,
                    f"Unknown load_in_low_bit value: {load_in_low_bit}, expected:"
                    f" sym_int4, asym_int4, sym_int5, asym_int5 or sym_int8.")
    qtype = ggml_tensor_qtype[load_in_low_bit]
    return ggml_convert_quant(model, qtype=qtype, optimize_model=optimize_atten)
