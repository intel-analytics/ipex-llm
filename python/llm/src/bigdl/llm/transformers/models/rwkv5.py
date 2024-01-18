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
# https://huggingface.co/RWKV/rwkv-5-world-3b/blob/main/modeling_rwkv5.py
# which is licensed under Apache License 2.0:
#
# Copyright 2023 Bo Peng and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
import torch.nn.functional as F

from typing import List


def rwkv_linear_attention_xpu(
    B: int,
    H: int,
    S: int,
    T: int,
    hidden: torch.Tensor,
    time_decay: torch.Tensor,
    time_first: torch.Tensor,
    receptance: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    lxw: torch.Tensor,
    lxb: torch.Tensor,
    ow: torch.nn.Linear,
    state: torch.Tensor,
):
    key = key.float().view(B, T, H, S).transpose(1, 2)
    value = value.float().view(B, T, H, S).transpose(1, 2)
    receptance = receptance.float().view(B, T, H, S).transpose(1, 2)

    time_decay = torch.exp(-torch.exp(time_decay.float()))
    time_first = time_first.float()

    state = state.contiguous().float()

    # `state` will be modified during this call
    import linear_q4_0
    out = linear_q4_0.rwkv_linear_attention_v5(
        time_decay,
        time_first,
        receptance,
        key,
        value,
        state,
    )

    lxw = lxw.float()
    lxb = lxb.float()

    out = out.reshape(B * T, H * S)
    out = F.group_norm(out, num_groups=H, weight=lxw, bias=lxb).reshape(B, T, H * S)
    out = out.to(dtype=hidden.dtype) * gate
    # out = out @ ow
    out = ow(out)
    return out, state


def rwkv_attention_forward(
    self,
    hidden: torch.Tensor,
    state: List[torch.Tensor]=None,
    use_cache: bool=False,
    seq_mode: bool=True,
):
    B = hidden.shape[0]
    H = self.time_decay.shape[0]
    S = hidden.shape[-1] // H
    T = hidden.shape[1]

    receptance, key, value, gate, state = self.extract_key_value(B, H, S, T, hidden, state=state)
    layer_state = state[1][:, :, :, :, self.layer_id] if state is not None else None

    if hidden.device.type == "xpu":
        rwkv, layer_state = rwkv_linear_attention_xpu(
            B,
            H,
            S,
            T,
            hidden,
            self.time_decay,
            self.time_faaaa,
            receptance,
            key,
            value,
            gate,
            self.ln_x.weight,
            self.ln_x.bias,
            self.output,
            state=layer_state,
        )
    else:
        from transformers.models.rwkv.modeling_rwkv import rwkv_linear_attention_cpu
        rwkv, layer_state = rwkv_linear_attention_cpu(
            B,
            H,
            S,
            T,
            self.num_attention_heads,
            hidden,
            self.time_decay,
            self.time_faaaa,
            receptance,
            key,
            value,
            gate,
            self.ln_x.weight,
            self.ln_x.bias,
            self.output.weight.t(),
            state=layer_state,
        )

    if layer_state is not None:
        state[1][:, :, :, :, self.layer_id] = layer_state

    return rwkv, state
