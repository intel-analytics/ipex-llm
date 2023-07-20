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
# and https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/
#     llama/modeling_llama.py
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
import torch.nn as nn
from accelerate import init_empty_weights
from bigdl.llm.transformers.linear_quant import LinearQuant, ParamsQuant
import warnings
from typing import List, Optional, Tuple, Union
import math
import torch.nn.functional as F
from bigdl.llm.utils.common import invalidInputError


def _replace_with_quant_linear(model, qtype, modules_to_not_convert=None,
                               current_key_name=None, convert_shape_only=False):
    has_been_replaced = False

    # Through our method, certain layers that were initialized on the device "meta"
    # (associated with the lazy initialization strategy of low_cpu_mem_usage) are not
    # being correctly moved back to the CPU device for some reason. Therefore, we are
    # moving these layers back to the CPU here in order to prevent the occurrence
    # of NoImplementnError. Details refer to:
    # https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3110
    model_state_dict = model.state_dict()
    for name, param in model.named_parameters():
        if param.data.device == torch.device('meta'):
            from accelerate.utils.modeling import set_module_tensor_to_device
            param = model_state_dict[name]
            set_module_tensor_to_device(model,
                                        name,
                                        "cpu",
                                        torch.empty(*param.size(), dtype=torch.float32))

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []

        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
                with init_empty_weights():

                    new_linear = LinearQuant(
                        module.in_features,
                        module.out_features,
                        qtype,
                        module.bias is not None,
                    )

                    # Copy the weights
                    paramsQuant = ParamsQuant(data=module.weight.data,
                                              requires_grad=False,
                                              quantized=False,
                                              convert_shape_only=convert_shape_only,
                                              _shape=None,
                                              qtype=qtype).to("cpu")
                    new_linear._parameters['weight'] = paramsQuant

                    if module.bias is not None:
                        new_linear._parameters['bias'] = nn.Parameter(module.bias.data).to("cpu")

                    model._modules[name] = new_linear
                    has_been_replaced = True
                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)

                    module.weight = None

        # Remove the last key for recursion
        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_quant_linear(
                module,
                qtype,
                modules_to_not_convert,
                current_key_name,
                convert_shape_only,
            )
    return model, has_been_replaced


def ggml_convert_quant(model, qtype, convert_shape_only=False):
    modules_to_not_convert = []  # ["lm_head"]
    model, has_been_replaced = _replace_with_quant_linear(
        model, qtype, modules_to_not_convert, None, convert_shape_only=convert_shape_only
    )
    if not has_been_replaced:
        warnings.warn(
            "No linear modules were found in "
            "your model. This can happen for some architectures such as gpt2 that uses Conv1D "
            "instead of Linear layers. Please double check your model architecture, or submit "
            "an issue on github if you think this is a bug."
        )
    else:
        model.to(torch.float32)
    return model


def convert_forward(m, target_m, new_forward):
    for _, sub_m in m.named_children():
        if isinstance(sub_m, target_m):
            bound_method = new_forward.__get__(sub_m, sub_m.__class__)
            setattr(sub_m, "forward", bound_method)
        convert_forward(sub_m, target_m, new_forward)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states
    go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads,
                                                           n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


KV_CACHE_ALLOC_BLOCK_LENGTH = 256


def llama_attention_forward_4_31(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if self.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
        query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim)
                                                // self.pretraining_tp, dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i])
                        for i in range(self.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i])
                      for i in range(self.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i])
                        for i in range(self.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len,
                                     self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len,
                                 self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len,
                                     self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        # key_states = torch.cat([past_key_value[0], key_states], dim=2)
        # value_states = torch.cat([past_key_value[1], value_states], dim=2)
        if kv_seq_len > self.max_cache_length:
            new_cache_key = torch.empty(bsz, self.num_heads,
                                        kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH, self.head_dim)
            new_cache_key[:, :, :kv_seq_len-1, :] = self.kv_cache[0][:, :, :kv_seq_len-1, :]
            self.kv_cache[0] = new_cache_key

            new_cache_value = torch.empty(bsz, self.num_heads,
                                          kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH, self.head_dim)
            new_cache_value[:, :, :kv_seq_len-1, :] = self.kv_cache[1][:, :, :kv_seq_len-1, :]
            self.kv_cache[1] = new_cache_value

        self.kv_cache[0][:, :, kv_seq_len-1:kv_seq_len, :] = key_states
        self.kv_cache[1][:, :, kv_seq_len-1:kv_seq_len, :] = value_states
        key_states = self.kv_cache[0][:, :, :kv_seq_len, :]
        value_states = self.kv_cache[1][:, :, :kv_seq_len, :]
    elif use_cache:
        # first token case
        self.max_cache_length = max(min(self.max_position_embeddings, 2 * kv_seq_len),
                                    kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH)
        self.kv_cache = (torch.empty(bsz, self.num_heads, self.max_cache_length, self.head_dim),
                         torch.empty(bsz, self.num_heads, self.max_cache_length, self.head_dim))
        self.kv_cache[0][:, :, :kv_seq_len, :] = key_states
        self.kv_cache[1][:, :, :kv_seq_len, :] = value_states

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states,
                                key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        invalidInputError(False,
                          f"Attention weights should be of size {(bsz, self.num_heads,
                                                                  q_len, kv_seq_len)}, "
                          f"but is {attn_weights.size()}")

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            invalidInputError(False,
                              f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, "
                              "but is {attention_mask.size()}")
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                         dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        invalidInputError(False,
                          f"attn_output should be of size {(bsz, self.num_heads,
                                                              q_len, self.head_dim)}, "
                          f"but is {attn_output.size()}")

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i])
                           for i in range(self.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
