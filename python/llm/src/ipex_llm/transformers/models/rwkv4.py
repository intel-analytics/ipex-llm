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
# https://github.com/huggingface/transformers/blob/v4.36.0/src/transformers/models/rwkv/modeling_rwkv.py
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

from typing import List


def extract_key_value(self, hidden, state=None):
    # Mix hidden with the previous timestep to produce key, value, receptance
    if hidden.size(1) == 1 and state is not None:
        shifted = state[1][:, :, self.layer_id]
    else:
        shifted = self.time_shift(hidden)
        if state is not None:
            shifted[:, 0] = state[1][:, :, self.layer_id]
    if len(shifted.size()) == 2:
        shifted = shifted.unsqueeze(1)
    shifted = shifted.contiguous()

    if not hasattr(self, "mixed_mix"):
        self.mixed_mix = torch.cat([
            self.time_mix_key.data,
            self.time_mix_value.data,
            self.time_mix_receptance.data,
        ]).to(dtype=hidden.dtype)

    import xe_linear
    mixed_result = xe_linear.rwkv_time_shift(hidden, shifted, self.mixed_mix)
    key, value, receptance = mixed_result

    key = self.key(key)
    value = self.value(value)
    receptance = torch.sigmoid(self.receptance(receptance))
    if state is not None:
        state[1][:, :, self.layer_id] = hidden[:, -1]
    return receptance, key, value, state


def rwkv_linear_attention_xpu(
    time_decay: torch.Tensor,
    time_first: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: List[torch.Tensor]=None,
    return_state: bool=False
):
    if state is None:
        num_state = torch.zeros(key.size(0), key.size(-1),
                                dtype=key.dtype, device=key.device)
        den_state = torch.zeros(key.size(0), key.size(-1),
                                dtype=key.dtype, device=key.device)
        max_state = torch.zeros(key.size(0), key.size(-1),
                                dtype=key.dtype, device=key.device) - 1e38
    else:
        num_state, den_state, max_state = state
        num_state = num_state.contiguous()
        den_state = den_state.contiguous()
        max_state = max_state.contiguous()

    time_decay = -torch.exp(time_decay)

    # `num_state`, `den_state`, `max_state` will be modified during this call
    import xe_linear
    output = xe_linear.rwkv_linear_attention_v4(
        time_decay,
        time_first,
        key,
        value,
        num_state,
        den_state,
        max_state,
    )

    if return_state or state is not None:
        state = [num_state, den_state, max_state]

    return output, state


def rwkv_attention_forward(
    self,
    hidden: torch.Tensor,
    state: List[torch.Tensor]=None,
    use_cache: bool=False,
):
    receptance, key, value, state = extract_key_value(self, hidden, state=state)
    layer_state = tuple(s[:, :, self.layer_id] for s in state[2:]) if state is not None else None

    if hidden.device.type == "xpu":
        self.time_decay.data = self.time_decay.data.to(dtype=key.dtype)
        self.time_first.data = self.time_first.data.to(dtype=key.dtype)
        rwkv, layer_state = rwkv_linear_attention_xpu(
            self.time_decay,
            self.time_first,
            key,
            value,
            state=layer_state,
            return_state=use_cache,
        )
    else:
        from transformers.models.rwkv.modeling_rwkv import rwkv_linear_attention_cpu
        rwkv, layer_state = rwkv_linear_attention_cpu(
            self.time_decay,
            self.time_first,
            key,
            value,
            state=layer_state,
            return_state=use_cache,
        )

    if layer_state is not None:
        state[2][:, :, self.layer_id] = layer_state[0]
        state[3][:, :, self.layer_id] = layer_state[1]
        state[4][:, :, self.layer_id] = layer_state[2]

    return self.output(receptance * rwkv), state


def rwkv_ffn_forward(
    self,
    hidden: torch.Tensor,
    state: List[torch.Tensor]=None,
):
    if hidden.size(1) == 1 and state is not None:
        shifted = state[0][:, :, self.layer_id]
    else:
        shifted = self.time_shift(hidden)
        if state is not None:
            shifted[:, 0] = state[0][:, :, self.layer_id]
    if len(shifted.size()) == 2:
        shifted = shifted.unsqueeze(1)
    shifted = shifted.contiguous()

    if not hasattr(self, "mixed_mix"):
        self.mixed_mix = torch.cat([self.time_mix_key.data,
                                    self.time_mix_receptance.data]).to(dtype=hidden.dtype)

    import xe_linear
    mixed_result = xe_linear.rwkv_time_shift(hidden, shifted, self.mixed_mix)
    key, receptance = mixed_result

    key = torch.square(torch.relu(self.key(key)))
    value = self.value(key)
    receptance = torch.sigmoid(self.receptance(receptance))

    if state is not None:
        state[0][:, :, self.layer_id] = hidden[:, -1]

    return receptance * value, state
