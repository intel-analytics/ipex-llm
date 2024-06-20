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
from typing import Union, Callable, Any
from typing import List
import torch
from ipex_llm.transformers.convert import convert_forward
from ipex_llm.utils.common.log4Error import invalidInputError

def compile(
    model: torch.nn.Module,
    dtype: torch.dtype = torch.float16,
    kv_cache_len_max: int = 1024
) -> torch.nn.Module:
    
    invalidInputError(dtype == torch.float16,
                      f"only torch.float16 is supported, but got {dtype}")
    
    if 'llama' in model.config.model_type:
        from ipex_llm.transformers.npu.llama import offload_llama_decoder_to_npu
        num_layers = model.config.num_hidden_layers
        model = offload_llama_decoder_to_npu(model, num_layers=num_layers,
                                             kv_cache_len_max=kv_cache_len_max)
    else:
        invalidInputError(f"model type not supported: {model.config.model_type}")
    
    return model
