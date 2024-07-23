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
# This file is adapted from
# https://github.com/FasterDecoding/SnapKV/blob/main/snapkv/monkeypatch/snapkv_utils.py
#

import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
import os
from ipex_llm.utils.common.log4Error import invalidInputError


# Copied from transformers.models.llama.modeling_llama.repeat_kv
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

class SnapKVCluster():
    def __init__(self, window_size=64, max_capacity_prompt=256 + 64, kernel_size=5,
                 pooling='avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def reset(self, window_size=64, max_capacity_prompt=256 + 64, kernel_size=5,
              pooling='avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, key_states, query_states, value_states, attention_mask,
                  num_key_value_groups):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            key_states_expand = repeat_kv(key_states, num_key_value_groups).to(key_states.device)
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :],
                                        key_states_expand.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min,
                              device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                                 dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim=-2)
            if self.pooling == 'avgpool':
                if num_key_value_groups > 1:
                    attn_cache = F.avg_pool2d(attn_weights_sum, kernel_size=(num_key_value_groups,
                                                                             self.kernel_size),
                                              padding=(0, self.kernel_size//2),
                                              stride=(num_key_value_groups, 1))
                else:
                    attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=self.kernel_size,
                                              padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                if num_key_value_groups > 1:
                    attn_cache = F.max_pool2d(attn_weights_sum, kernel_size=(num_key_value_groups,
                                                                             self.kernel_size),
                                              padding=(0, self.kernel_size//2),
                                              stride=(num_key_value_groups, 1))
                else:
                    attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=self.kernel_size,
                                              padding=self.kernel_size//2, stride=1)
            else:
                invalidInputError(False, 'Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2,
                                                                               index=indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states


def init_snapkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        self.kv_cluster = SnapKVCluster( 
            window_size=self.config.window_size, 
            max_capacity_prompt=self.config.max_capacity_prompt, 
            kernel_size=self.config.kernel_size,
            pooling=self.config.pooling
        )


def should_use_snapkv():
    use_snap_kv = os.environ.get("IPEX_LLM_COMPRESS_KV_CACHE", None)
    return use_snap_kv == "1"
