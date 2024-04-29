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


import torch

from .models.utils import (
    init_fp8_kv_cache, append_fp8_kv_cache,
    init_kv_cache, append_kv_cache
)
from typing import Optional, Dict, Tuple, Any
from transformers.cache_utils import DynamicCache


class DynamicFp8Cache(DynamicCache):
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]]=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, num_heads, seq_len, head_dim = key_states.shape

        if layer_idx == 0:
            if hasattr(self, "_seen_tokens"):
                # 4.39 uses `_seen_tokens`
                self._seen_tokens += seq_len
            else:
                # 4.37 uses `seen_tokens`
                self.seen_tokens += seq_len

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            k_cache, v_cache = init_fp8_kv_cache(
                batch_size, num_heads, seq_len, head_dim,
                device=key_states.device,
            )
            k_cache, v_cache = append_fp8_kv_cache(k_cache, v_cache, key_states, value_states)

            self.key_cache.append(k_cache)
            self.value_cache.append(v_cache)
        else:
            k_cache = self.key_cache[layer_idx]
            v_cache = self.value_cache[layer_idx]
            k_cache, v_cache = append_fp8_kv_cache(k_cache, v_cache, key_states, value_states)
            self.key_cache[layer_idx] = k_cache
            self.value_cache[layer_idx] = v_cache

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class DynamicNormalCache(DynamicCache):
    KV_ALLOC_BLOCK_LENGTH = 256

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]]=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, num_heads, seq_len, head_dim = key_states.shape

        if layer_idx == 0:
            if hasattr(self, "_seen_tokens"):
                # 4.39 uses `_seen_tokens`
                self._seen_tokens += seq_len
            else:
                # 4.37 uses `seen_tokens`
                self.seen_tokens += seq_len

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            k_cache, v_cache = init_kv_cache(
                batch_size, num_heads, head_dim,
                0, key_states.size(2) + self.KV_ALLOC_BLOCK_LENGTH,
                key_states.dtype, key_states.device
            )
            k_cache, v_cache = append_kv_cache(k_cache, v_cache, key_states, value_states)

            self.key_cache.append(k_cache)
            self.value_cache.append(v_cache)
        else:
            k_cache = self.key_cache[layer_idx]
            v_cache = self.value_cache[layer_idx]

            kv_seq_len = k_cache.size(2) + key_states.size(2)
            if k_cache.stride(1) < kv_seq_len * k_cache.size(3):
                new_k_cache, new_v_cache = init_kv_cache(
                    batch_size, num_heads, head_dim,
                    k_cache.size(2), kv_seq_len + self.KV_ALLOC_BLOCK_LENGTH,
                    key_states.dtype, key_states.device
                )
                new_k_cache[...] = k_cache[...]
                new_v_cache[...] = v_cache[...]
                k_cache = new_k_cache
                v_cache = new_v_cache
            k_cache, v_cache = append_kv_cache(k_cache, v_cache, key_states, value_states)
            self.key_cache[layer_idx] = k_cache
            self.value_cache[layer_idx] = v_cache

        return self.key_cache[layer_idx], self.value_cache[layer_idx]
