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
from typing import Optional, Dict, Tuple, Any
from transformers.cache_utils import DynamicCache
import sys


def init_fused_kv_cache(
    batch_size, num_heads, head_dim, current_length, max_length, dtype, device, tranpose_value=False
):
    if not tranpose_value:
        key_cache_storage = torch.zeros(
            batch_size, num_heads, max_length, head_dim, dtype=dtype, device=device
        )
        value_cache_storage = torch.zeros(
            batch_size, num_heads, max_length, head_dim, dtype=dtype, device=device
        )

        key_cache = key_cache_storage.as_strided(
            (batch_size, num_heads, current_length, head_dim),
            key_cache_storage.stride(),
            storage_offset=0,
        )
        value_cache = value_cache_storage.as_strided(
            (batch_size, num_heads, current_length, head_dim),
            value_cache_storage.stride(),
            storage_offset=0,
        )
        return key_cache, value_cache
    else:
        key_cache_storage = torch.zeros(
            batch_size, num_heads, max_length, head_dim, dtype=dtype, device=device
        )
        value_cache_storage = torch.zeros(
            batch_size, num_heads, head_dim, max_length, dtype=dtype, device=device
        )

        key_cache = key_cache_storage.as_strided(
            (batch_size, num_heads, current_length, head_dim),
            key_cache_storage.stride(),
            storage_offset=0,
        )
        value_cache = value_cache_storage.as_strided(
            (batch_size, num_heads, head_dim, current_length),
            value_cache_storage.stride(),
            storage_offset=0,
        )
        return key_cache, value_cache.transpose(-1, -2)


def append_fused_kv_cache(cache_k, cache_v, key_states, value_states, transpose_value=False):
    if not transpose_value:
        new_size = (
            cache_k.size(0),
            cache_k.size(1),
            cache_k.size(2) + key_states.size(2),
            cache_k.size(3),
        )
        new_cache_k = cache_k.as_strided(new_size, cache_k.stride(), storage_offset=0)
        new_cache_k[:, :, cache_k.size(2):cache_k.size(2) + key_states.size(2), :] = key_states
        new_cache_v = cache_v.as_strided(new_size, cache_v.stride(), storage_offset=0)
        new_cache_v[:, :, cache_v.size(2):cache_v.size(2) + key_states.size(2), :] = value_states
        return new_cache_k, new_cache_v
    else:
        new_size_key = (
            cache_k.size(0),
            cache_k.size(1),
            cache_k.size(2) + key_states.size(2),
            cache_k.size(3),
        )
        new_cache_k = cache_k.as_strided(new_size_key, cache_k.stride(), storage_offset=0)
        new_cache_k[:, :, cache_k.size(2):cache_k.size(2) + key_states.size(2), :] = key_states

        new_size_value = (
            cache_v.size(0),
            cache_v.size(1),
            cache_v.size(3),
            cache_v.size(2) + value_states.size(3),
        )
        raw_cache_v = cache_v.transpose(-1, -2)
        new_cache_v = raw_cache_v.as_strided(new_size_value, raw_cache_v.stride(), storage_offset=0)
        start = raw_cache_v.size(3)
        end = raw_cache_v.size(3) + value_states.size(3)
        new_cache_v[:, :, :, start:end] = value_states
        return new_cache_k, new_cache_v.transpose(-1, -2)


def expand_fused_kv_cache(cache_k, cache_v, transpose_value=False):
    if not transpose_value:
        new_size = (cache_k.size(0), cache_k.size(1), cache_k.size(2) + 1, cache_k.size(3))
        new_cache_k = cache_k.as_strided(new_size, cache_k.stride(), storage_offset=0)
        new_cache_v = cache_v.as_strided(new_size, cache_v.stride(), storage_offset=0)
        return new_cache_k, new_cache_v
    else:
        new_size_key = (cache_k.size(0), cache_k.size(1), cache_k.size(2) + 1, cache_k.size(3))
        new_cache_k = cache_k.as_strided(new_size_key, cache_k.stride(), storage_offset=0)
        new_size_value = (
            cache_v.size(0),
            cache_v.size(1),
            cache_v.size(3),
            cache_v.size(2) + 1,
        )
        raw_cache_v = cache_v.transpose(-1, -2)
        new_cache_v = raw_cache_v.as_strided(new_size_value, raw_cache_v.stride(), storage_offset=0)
        return new_cache_k, new_cache_v.transpose(-1, -2)


def shrink_fused_kv_cache(cache_k, cache_v, new_seq_len, transpose_value=False):
    if not transpose_value:
        new_size = (cache_k.size(0), cache_k.size(1), new_seq_len, cache_k.size(3))
        new_cache_k = cache_k.as_strided(new_size, cache_k.stride(), storage_offset=0)
        new_cache_v = cache_v.as_strided(new_size, cache_v.stride(), storage_offset=0)
        return new_cache_k, new_cache_v
    else:
        new_size_key = (cache_k.size(0), cache_k.size(1), new_seq_len, cache_k.size(3))
        new_cache_k = cache_k.as_strided(new_size_key, cache_k.stride(), storage_offset=0)
        new_size_value = (
            cache_v.size(0),
            cache_v.size(1),
            cache_v.size(3),
            new_seq_len,
        )
        raw_cache_v = cache_v.transpose(-1, -2)
        new_cache_v = raw_cache_v.as_strided(new_size_value, raw_cache_v.stride(), storage_offset=0)
        return new_cache_k, new_cache_v.transpose(-1, -2)


class DynamicFusedNormalCache(DynamicCache):
    # Experimental support for fused decoderlayer implementation on NPU
    # Currently only for llama2

    def __init__(self) -> None:
        self.key_cache: Dict[int, torch.Tensor] = {}
        self.value_cache: Dict[int, torch.Tensor] = {}
        self.min_layer_idx = sys.maxsize

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]]=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, num_heads, seq_len, head_dim = key_states.shape

        max_seq_length = cache_kwargs["max_seq_len"] if "max_seq_len" in cache_kwargs else None
        transpose_value = cache_kwargs["transpose"] if "transpose" in cache_kwargs else False

        # Update the cache
        # if len(self.key_cache) <= layer_idx:
        if layer_idx not in self.key_cache:
            max_len = max_seq_length
            k_cache, v_cache = init_fused_kv_cache(
                batch_size,
                num_heads,
                head_dim,
                0,
                max_len,
                # key_states.dtype,
                torch.float16,
                key_states.device,
                tranpose_value=transpose_value,
            )
            k_cache, v_cache = append_fused_kv_cache(
                k_cache, v_cache, key_states, value_states, transpose_value=transpose_value
            )

            self.key_cache[layer_idx] = k_cache
            self.value_cache[layer_idx] = v_cache
        else:
            k_cache = self.key_cache[layer_idx]
            v_cache = self.value_cache[layer_idx]

            kv_seq_len = k_cache.size(2) + key_states.size(2)
            k_cache, v_cache = append_fused_kv_cache(
                k_cache, v_cache, key_states, value_states, transpose_value=transpose_value
            )
            self.key_cache[layer_idx] = k_cache
            self.value_cache[layer_idx] = v_cache

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states.
        A layer index can be optionally passed."""

        for idx, layer in self.key_cache.items():
            return layer.shape[-2]
        return 0

    def expand(self, transpose_value=True):
        for idx, layer in self.key_cache.items():
            key_cache, value_cache = expand_fused_kv_cache(
                self.key_cache[idx],
                self.value_cache[idx],
                transpose_value=transpose_value,
            )
            self.key_cache[idx] = key_cache
            self.value_cache[idx] = value_cache

    def shrink(self, new_seq_len, transpose_value=True):
        for idx, layer in self.key_cache.items():
            key_cache, value_cache = shrink_fused_kv_cache(
                self.key_cache[idx], self.value_cache[idx], new_seq_len, transpose_value
            )
            self.key_cache[idx] = key_cache
            self.value_cache[idx] = value_cache

    @property
    def _seen_tokens(self):
        return self.get_seq_length()

    @property
    def seen_tokens(self):
        return self.get_seq_length()
