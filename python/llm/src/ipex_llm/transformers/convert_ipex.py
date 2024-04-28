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
from ipex_llm.utils.common import invalidInputError
from typing import List, Optional, Tuple, Union
from intel_extension_for_pytorch.transformers.optimize import (
    lowering_class_cpu,
    convert_class,
)
from intel_extension_for_pytorch.cpu._auto_kernel_selection import (
    _enable_tpp,
    _using_tpp,
    _disable_tpp
)
from ipex_llm.ggml.quantize import ggml_tensor_qtype
from ipex_llm.transformers.convert import get_enable_ipex
import os


def _ipex_optimize_rmsnorm(_model, supported_classes, is_tpp=False, is_woq=False):
    from intel_extension_for_pytorch.transformers.models.cpu.fusions.mha_fusion import _IPEXRMSNorm
    for supported_class in supported_classes:
        lowering_class_cpu(
            _model,
            supported_class,
            _IPEXRMSNorm,
            _model.config,
            tpp=is_tpp,
            woq=is_woq,
        )


def _ipex_optimize_decoder(model, is_tpp=False, is_woq=False):
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
            tpp=is_tpp,
            woq=is_woq,
        )


def _ipex_optimize_attention(model, is_tpp=False, is_woq=False):
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
            tpp=is_tpp,
            woq=is_woq,
        )


def _ipex_optimize_model(model, rms_classes, qtype):
    import intel_extension_for_pytorch as ipex
    from intel_extension_for_pytorch.transformers.models.reference.models import output_hook
    from intel_extension_for_pytorch.transformers.optimize import ipex_quantization_flow

    is_woq = False
    is_quantization = False
    _disable_tpp()
    if qtype == ggml_tensor_qtype["bf16"]:
        _enable_tpp()
        model = ipex.optimize(model.eval(), dtype=torch.bfloat16, inplace=True).eval()
    elif qtype == ggml_tensor_qtype["sym_int4"]:
        is_quantization = True
        is_woq = True
        act_quant_mode_dict = {
            "PER_TENSOR": ipex.quantization.WoqActQuantMode.PER_TENSOR,
            "PER_IC_BLOCK": ipex.quantization.WoqActQuantMode.PER_IC_BLOCK,
            "PER_BATCH": ipex.quantization.WoqActQuantMode.PER_BATCH,
            "PER_BATCH_IC_BLOCK": ipex.quantization.WoqActQuantMode.PER_BATCH_IC_BLOCK,
        }
        qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
            weight_dtype=torch.quint4x2,  # INT4
            lowp_mode=ipex.quantization.WoqLowpMode.INT8,
            act_quant_mode=act_quant_mode_dict["PER_IC_BLOCK"],
            group_size=-1,
        )
        model = ipex_quantization_flow(model, torch.bfloat16, None, qconfig, None)
    elif qtype == ggml_tensor_qtype["sym_int8"]:
        is_quantization = True
        is_woq = True
        act_quant_mode_dict = {
            "PER_TENSOR": ipex.quantization.WoqActQuantMode.PER_TENSOR,
            "PER_IC_BLOCK": ipex.quantization.WoqActQuantMode.PER_IC_BLOCK,
            "PER_BATCH": ipex.quantization.WoqActQuantMode.PER_BATCH,
            "PER_BATCH_IC_BLOCK": ipex.quantization.WoqActQuantMode.PER_BATCH_IC_BLOCK,
        }
        qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
            weight_dtype=torch.qint8,  # INT8
            lowp_mode=ipex.quantization.WoqLowpMode.BF16,
            act_quant_mode=act_quant_mode_dict["PER_IC_BLOCK"],
            group_size=-1,
        )
        model = ipex_quantization_flow(model, torch.bfloat16, None, qconfig, None)

    is_tpp = _using_tpp()

    _ipex_optimize_rmsnorm(model, rms_classes, is_tpp=is_tpp, is_woq=is_woq)
    _ipex_optimize_attention(model, is_tpp=is_tpp, is_woq=is_woq)
    _ipex_optimize_decoder(model, is_tpp=is_tpp, is_woq=is_woq)

    # need to register_forward_hook after torch.jit.trace
    # model.register_forward_hook(output_hook, with_kwargs=True)
    return model


def _ipex_jit(model):
    from intel_extension_for_pytorch.transformers.optimize import (
        get_dummy_input,
        _set_optimized_model_for_generation
    )
    sample_inputs = (
        get_dummy_input(model, return_dict=True)
    )
    if "return_last_logit" in sample_inputs:
        sample_inputs["return_last_logit"] = torch.tensor(False)
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
    from intel_extension_for_pytorch.transformers.models.reference.models import output_hook
    model.register_forward_hook(output_hook, with_kwargs=True)

    return model


def convert_function(m, func_name, new_function):
    bound_method = new_function.__get__(m, m.__class__)
    setattr(m, func_name, bound_method)


def GLM_get_masks(self, input_ids, past_key_values, padding_mask=None):
    batch_size, seq_length = input_ids.shape
    full_attention_mask = torch.ones(
        batch_size, seq_length, seq_length, device=input_ids.device
    )
    full_attention_mask.tril_()
    past_length = 0
    if past_key_values:
        if len(past_key_values[0]) != 4:  # not discrete kv cache
            past_length = past_key_values[0][0].shape[0]
        else:  # discrete kv cache
            past_length = past_key_values[0][0].shape[-2]

    _enable_ipex = get_enable_ipex()
    # always call for jit
    if past_length or _enable_ipex:
        full_attention_mask = torch.cat(
            (
                torch.ones(
                    batch_size, seq_length, past_length, device=input_ids.device
                ),
                full_attention_mask,
            ),
            dim=-1,
        )
    if padding_mask is not None:
        full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
    # if not past_length and padding_mask is not None:
    #     full_attention_mask -= padding_mask.unsqueeze(-1) - 1
    full_attention_mask = (full_attention_mask < 0.5).bool()
    full_attention_mask.unsqueeze_(1)
    return full_attention_mask


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

    _enable_ipex = get_enable_ipex()
    if _enable_ipex or past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)  # noqa

    # add lower triangular sliding window mask if necessary
    if sliding_window is not None:
        diagonal = past_key_values_length - sliding_window + 1

        context_mask = 1 - torch.triu(torch.ones_like(mask, dtype=torch.int), diagonal=diagonal)
        mask.masked_fill_(context_mask.bool(), torch.finfo(dtype).min)

    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask


def _llama_model_forward_4_35(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions  # noqa
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states  # noqa
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")  # noqa
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")  # noqa

    past_key_values_length = 0
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device  # noqa
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if getattr(self.config, "_flash_attn_2_enabled", False):
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None  # noqa
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

    # embed positions
    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."  # noqa
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)  # noqa
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
