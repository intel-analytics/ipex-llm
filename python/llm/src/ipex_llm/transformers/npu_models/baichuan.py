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
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llama/modeling_llama.py
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
from torch.nn import functional as F
import importlib
from typing import Optional, Tuple
from ipex_llm.transformers.npu_models.common import merge_linear
from ipex_llm.transformers.models.utils import update_past_key_value


def merge_mlp(module: torch.nn.Module):
    if type(module).__name__ == "MLP":
        gate_up_proj = merge_linear([
            module.gate_proj,
            module.up_proj,
        ])
        module.gate_up_proj = gate_up_proj
        del module.gate_proj, module.up_proj


def baichuan_mlp_forward(self, x):
    gate_up_proj = self.gate_up_proj(x)
    gate_proj, up_proj = gate_up_proj.chunk(2, dim=-1)
    down_proj = self.down_proj(self.act_fn(gate_proj) * up_proj)
    return down_proj


def baichuan_attention_fwd(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    modeling_module_name = self.__class__.__module__
    module = importlib.import_module(modeling_module_name)
    apply_rotary_pos_emb = module.apply_rotary_pos_emb

    bsz, q_len, _ = hidden_states.size()

    proj = self.W_pack(hidden_states)
    proj = proj.unflatten(-1, (3, self.hidden_size)).unsqueeze(0).transpose(0, -2).squeeze(-2)
    query_states = proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)
    # [bsz, nh, t, hd]

    key_states, value_states = update_past_key_value(
        past_key_value, key_states, value_states,
        kv_seq_len, False, "cpu"
    )

    past_key_value = (key_states, value_states) if use_cache else None

    if query_states.size(2) == key_states.size(2):
        # first token
        from intel_npu_acceleration_library.functional import scaled_dot_product_attention
        attn_output = scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask
        )
        attn_weights = None
    else:
        with torch.backends.cuda.sdp_kernel(enable_flash=True,
                                            enable_math=True, enable_mem_efficient=True):
            attn_output = F.scaled_dot_product_attention(query_states, key_states,
                                                         value_states, attn_mask=attention_mask)
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
