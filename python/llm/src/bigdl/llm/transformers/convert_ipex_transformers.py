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


# Some parts of this file is adapted from
# https://github.com/huggingface/transformers/blob/v4.30.2/src/transformers/utils/bitsandbytes.py
# and https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
# which is licensed under Apache License 2.0:
#
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

import os
import torch.nn as nn
from accelerate import init_empty_weights

from bigdl.llm.utils.common.log4Error import invalidInputError
from bigdl.llm.transformers.linear_quant import LinearQuant, ParamsQuant

from intel_extension_for_pytorch.nn.utils._transformers import IPEXEmptyLinear


def _convert_ipex_transformers(model: nn.Module):
    for name, module in model.named_children():
        if isinstance(module, (nn.Linear, IPEXEmptyLinear)):
            with init_empty_weights():
                # create new quantized linear
                if isinstance(module, nn.Linear):
                    new_linear = LinearQuant(
                        module.in_features,
                        module.out_features,
                        2,
                        module.bias is not None,
                    )
                else:
                    # "out_proj" in Llama cannot be converted to int4
                    if "out_proj" in name or module.weight is None:
                        continue

                    new_linear = LinearQuant(
                        module.weight.data.shape[1],
                        module.weight.data.shape[0],
                        2,
                        module.bias is not None,
                    )
                # copy weights and bias
                paramsQuant = ParamsQuant(data=module.weight.data,
                                          requires_grad=False,
                                          quantized=False,
                                          qtype=2).to("cpu")
                new_linear._parameters['weight'] = paramsQuant.to("xpu")
                new_linear._parameters['bias'] = module.bias
                module.weight = None
                module.bias = None

                # replace linear
                model._modules[name] = new_linear
                model._modules[name].requires_grad_(False)
        elif len(list(module.children())) > 0:
            _convert_ipex_transformers(module)

    return model


def convert_ipex_transformers(model: nn.Module, qtype: int = 2):
    from transformers import LlamaForCausalLM
    invalidInputError(isinstance(model, LlamaForCausalLM),
                      "Now only `LlamaForCausalLM` is supported.")
    invalidInputError(qtype == 2,
                      "Now only q4_0(qtype=2) is supported.")
    col_major = os.environ.get("COL_MAJOR", "OFF").upper()in ["1", "Y", "ON", "YES", "TRUE"]
    invalidInputError(col_major, 'Now only col_major mode is supported. (export COL_MAJOR=1)')
    return _convert_ipex_transformers(model)
