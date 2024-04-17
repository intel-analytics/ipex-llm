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
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py

# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

""" PyTorch Phixtral model."""
import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from ipex_llm.ggml.quantize import ggml_tensor_qtype
from ipex_llm.utils.common import invalidInputError
from ipex_llm.transformers.models.utils import init_kv_cache, extend_kv_cache, append_kv_cache
from ipex_llm.transformers.models.utils import apply_rotary_pos_emb,\
    apply_rotary_pos_emb_no_cache_xpu, is_enough_kv_cache_room_4_36
from ipex_llm.transformers.models.mistral import should_use_fuse_rope
from ipex_llm.transformers.models.utils import use_flash_attention
from ipex_llm.transformers.models.utils import mlp_fusion_check

import os

KV_CACHE_ALLOC_BLOCK_LENGTH = int(os.environ.get("KV_CACHE_ALLOC_BLOCK_LENGTH", 256))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads,
                                                           n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def phixtral_moeblock_forward(self, hidden_states: torch.Tensor):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    bs = hidden_states.shape[0]
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    num_local_experts = len(self.mlp)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    top_k = self.num_experts_per_tok
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    if bs > 1:
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts,
                                                  num_classes=num_local_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(num_local_experts):
            expert_layer = self.mlp[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state)

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    else:
        selected_experts = selected_experts[0].cpu().tolist()
        for idx in range(top_k):
            exp_id = selected_experts[idx]
            expert_layer = self.mlp[exp_id]
            weight = routing_weights[:, idx]
            if idx == 0:
                final_hidden_states = expert_layer(hidden_states)
            else:
                final_hidden_states = final_hidden_states + expert_layer(hidden_states)

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states


def phixtral_mlp_forward(
    self,
    x: torch.Tensor,
) -> torch.Tensor:
    hidden_states = self.fc1(x)
    hidden_states = self.act(hidden_states)
    hidden_states = self.fc2(hidden_states)

    return hidden_states
