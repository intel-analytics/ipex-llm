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

from typing import List, Optional


def extract_key_value(self, hidden, state=None):
    # Mix hidden with the previous timestep to produce key, value, receptance
    if hidden.size(1) == 1 and state is not None:
        shifted = state[0][self.layer_id]
    else:
        shifted = self.time_shift(hidden)
        if state is not None:
            shifted[:, 0] = state[0][self.layer_id]
    if len(shifted.size()) == 2:
        shifted = shifted.unsqueeze(1)

    if not hasattr(self, "mixed_mix"):
        self.mixed_mix = torch.cat([
            self.time_mix_key.data,
            self.time_mix_value.data,
            self.time_mix_receptance.data,
            self.time_mix_gate.data,
        ])

    import linear_q4_0
    mixed_result = linear_q4_0.rwkv_time_shift(hidden, shifted, self.mixed_mix)
    key, value, receptance, gate = mixed_result

    key = self.key(key)
    value = self.value(value)
    receptance = self.receptance(receptance)
    gate = F.silu(self.gate(gate))

    if state is not None:
        state[0][self.layer_id] = hidden[:, -1]

    return receptance, key, value, gate, state


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

    # `state` will be updated inplaced during this call
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
    return out


def rwkv_linear_attention_cpu(
    B,
    H,
    S,
    T,
    n_head,
    hidden,
    time_decay,
    time_first,
    receptance,
    key,
    value,
    gate,
    lxw,
    lxb,
    ow,
    state,
):
    key = key.to(torch.float32).view(B, T, H, S).transpose(1, 2).transpose(-2, -1)
    value = value.to(torch.float32).view(B, T, H, S).transpose(1, 2)
    receptance = receptance.to(torch.float32).view(B, T, H, S).transpose(1, 2)
    time_decay = torch.exp(-torch.exp(time_decay.float())).reshape(-1, 1, 1).reshape(n_head, -1, 1)
    time_first = time_first.float().reshape(-1, 1, 1).reshape(n_head, -1, 1)
    lxw = lxw.float()
    lxb = lxb.float()
    out = torch.zeros_like(key).reshape(B, T, H, S)
    for t in range(T):
        rt = receptance[:, :, t:t + 1, :]
        kt = key[:, :, :, t:t + 1]
        vt = value[:, :, t:t + 1, :]
        at = kt @ vt
        out[:, t] = (rt @ (time_first * at + state)).squeeze(2)
        with torch.no_grad():
            state = at + time_decay * state

    out = out.reshape(B * T, H * S)
    out = F.group_norm(out, num_groups=H, weight=lxw, bias=lxb).reshape(B, T, H * S)
    out = out.to(dtype=hidden.dtype) * gate
    # out = out @ ow
    out = ow(out)   # fix this

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

    if hidden.device.type == "xpu":
        receptance, key, value, gate, state = extract_key_value(self, hidden, state)
        # `state`` will be updated inplaced when running on GPU
        rwkv = rwkv_linear_attention_xpu(
            B, H, S, T,
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
            state=state[1][self.layer_id],
        )
    else:
        receptance, key, value, gate, state = self.extract_key_value(B, H, S, T, hidden, state)
        layer_state = state[1][:, :, :, :, self.layer_id] if state is not None else None
        rwkv, layer_state = rwkv_linear_attention_cpu(
            B, H, S, T,
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
            self.output,
            state=layer_state,
        )
        if layer_state is not None:
            state[1][:, :, :, :, self.layer_id] = layer_state

    return rwkv, state


def rwkv_ffn_forward_wrapper(origin_rwkv_ffn_forward):
    def rwkv_ffn_forward(
        self,
        hidden: torch.Tensor,
        state: List[torch.Tensor]=None,
    ):
        if hidden.device.type == "xpu":
            if hidden.size(1) == 1 and state is not None:
                shifted = state[2][self.layer_id]
            else:
                shifted = self.time_shift(hidden)
                if state is not None:
                    shifted[:, 0] = state[2][self.layer_id]
            if len(shifted.size()) == 2:
                shifted = shifted.unsqueeze(1)

            if not hasattr(self, "mixed_mix"):
                self.mixed_mix = torch.cat([self.time_mix_key.data, self.time_mix_receptance.data])

            import linear_q4_0
            mixed_result = linear_q4_0.rwkv_time_shift(hidden, shifted, self.mixed_mix)
            key, receptance = mixed_result

            key = torch.square(torch.relu(self.key(key)))
            value = self.value(key)
            receptance = torch.sigmoid(self.receptance(receptance))

            if state is not None:
                state[2][self.layer_id] = hidden[:, -1]

            return receptance * value, state
        else:
            return origin_rwkv_ffn_forward(self, hidden, state)

    return rwkv_ffn_forward


def rwkv_model_forward_wrapper(origin_rwkv_model_forward):
    def rwkv_model_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        state: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # change `state` layout and put `num_hidden_layers` to the highest dim
        if input_ids.device.type == "xpu" and use_cache and state is None:
            state = []
            batch_size = input_ids.size(0)
            hidden_size = self.config.hidden_size
            num_hidden_layers = self.config.num_hidden_layers
            num_attention_heads = self.config.hidden_size // self.config.num_attention_heads
            state.append(
                torch.zeros(
                    (num_hidden_layers, batch_size, hidden_size),
                    dtype=self.embeddings.weight.dtype,
                    requires_grad=False,
                    device=input_ids.device,
                ).contiguous()
            )
            state.append(
                torch.zeros(
                    (
                        num_hidden_layers,
                        batch_size,
                        num_attention_heads,
                        self.config.hidden_size // num_attention_heads,
                        self.config.hidden_size // num_attention_heads,
                    ),
                    dtype=torch.float32,
                    requires_grad=False,
                    device=input_ids.device,
                ).contiguous()
            )
            state.append(
                torch.zeros(
                    (num_hidden_layers, batch_size, hidden_size),
                    dtype=self.embeddings.weight.dtype,
                    requires_grad=False,
                    device=input_ids.device,
                ).contiguous()
            )
        return origin_rwkv_model_forward(
            self=self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            state=state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    return rwkv_model_forward
