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
# https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/llama/modeling_llama.py  # noqa
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

import torch
from bigdl.llm.utils.common import invalidInputError
from typing import List, Optional, Tuple, Union


def lowering_class_cpu(m, target_m, new_class, config, tpp=False, woq=False):
    for name, sub_m in m.named_children():
        if isinstance(sub_m, target_m):
            new_m = new_class(sub_m, config, tpp, woq)
            setattr(m, name, new_m)
        lowering_class_cpu(sub_m, target_m, new_class, config, tpp, woq)


def convert_class(m, target_m, new_class, config, distributed=False):
    for name, sub_m in m.named_children():
        if isinstance(sub_m, target_m):
            new_m = new_class(sub_m, config, distributed)
            setattr(m, name, new_m)
        convert_class(sub_m, target_m, new_class, config, distributed)


def _set_optimized_model_for_generation(
    model,
    optimized_model,
    first_token_optimized_model=None,
):
    from intel_extension_for_pytorch.transformers.models.reference.models import (
        IPEX_LLM_Model_Return
    )
    if first_token_optimized_model is not None:
        model.trace_graph_first = IPEX_LLM_Model_Return(
            model, first_token_optimized_model
        ).forward

    model.trace_graph = IPEX_LLM_Model_Return(model, optimized_model).forward
    print(
        "ipex.llm.optimize has set the optimized or quantization model for model.generate()"
    )
    return model


def _ipex_optimize_rmsnorm(_model):
    from intel_extension_for_pytorch.transformers.models.cpu.fusions.mha_fusion import _IPEXRMSNorm
    import transformers
    supported_classes = [
        transformers.models.llama.modeling_llama.LlamaRMSNorm,
    ]
    if _model.config.architectures[0] == "BaichuanForCausalLM":
        supported_classes.append(type(_model.model.layers[0].input_layernorm))
    if (
        _model.config.architectures[0] == "ChatGLMModel"
        and _model.config.rmsnorm
    ):
        supported_classes.append(
            type(_model.transformer.encoder.layers[0].input_layernorm)
        )
    for supported_class in supported_classes:
        lowering_class_cpu(
            _model,
            supported_class,
            _IPEXRMSNorm,
            _model.config,
            tpp=False,
            woq=False,
        )


def _ipex_optimize_decoder(model):
    from intel_extension_for_pytorch.transformers.models.reference.modules.decoder import (
        _IPEXDecoderLayerRef
    )
    from intel_extension_for_pytorch.transformers.models.cpu.modules.decoder import (
        _IPEXDecoderLayerCPU
    )
    for supported_mlp_class in [_IPEXDecoderLayerRef]:
        lowering_class_cpu(
            model,
            supported_mlp_class,
            _IPEXDecoderLayerCPU,
            model.config,
            tpp=False,
            woq=False,
        )


def _ipex_optimize_attention(model):
    from intel_extension_for_pytorch.transformers.models.reference.modules.attentions import (
        _IPEXAttentionRef
    )
    from intel_extension_for_pytorch.transformers.models.cpu.modules.attentions import (
        _IPEXAttentionCPU
    )
    for supported_mha_class in [_IPEXAttentionRef]:
        lowering_class_cpu(
            model,
            supported_mha_class,
            _IPEXAttentionCPU,
            model.config,
            tpp=False,
            woq=False,
        )


def _ipex_jit(model):
    from intel_extension_for_pytorch.transformers.optimize import get_dummy_input
    sample_inputs = (
        get_dummy_input(model, return_dict=True)
    )
    with torch.no_grad(), torch.cpu.amp.autocast(
        enabled=True
    ):
        trace_model = torch.jit.trace(
            model,
            example_kwarg_inputs=sample_inputs,
            strict=False,
            check_trace=False,
        )
        trace_model = torch.jit.freeze(trace_model)
        model = _set_optimized_model_for_generation(
            model, optimized_model=trace_model
        )

    return model.eval()


@staticmethod
def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
    sliding_window: Optional[int] = None,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

    import os
    _enable_ipex = os.getenv("BIGDL_OPT_IPEX")
    _enable_ipex = (_enable_ipex is not None) and (_enable_ipex.lower() == "true")
    if _enable_ipex or past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)  # noqa

    # add lower triangular sliding window mask if necessary
    if sliding_window is not None:
        diagonal = past_key_values_length - sliding_window + 1

        context_mask = 1 - torch.triu(torch.ones_like(mask, dtype=torch.int), diagonal=diagonal)
        mask.masked_fill_(context_mask.bool(), torch.finfo(dtype).min)

    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
