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
import torch.nn.functional as F
import torch.nn as nn
import math

from .models.utils import (
    init_fp8_kv_cache, append_fp8_kv_cache,
    init_kv_cache, append_kv_cache, extend_kv_cache
)
from typing import Optional, Dict, Tuple, Any, List
from transformers.cache_utils import DynamicCache
from ipex_llm.utils.common.log4Error import invalidInputError


class DynamicFp8Cache(DynamicCache):
    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        # ignore num_hidden_layers to fix transformers >= 4.45
        super().__init__()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]]=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # fix converting empty DynamicCache in transformers >= 4.45
        if key_states == []:
            return key_states, value_states

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

    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        # ignore num_hidden_layers to fix transformers >= 4.45
        super().__init__()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]]=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # fix converting empty DynamicCache in transformers >= 4.45
        if key_states == []:
            return key_states, value_states

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

    @classmethod
    def from_reserved(cls, layers: int,
                      bsz: int, n_head: int, length: int, head_dim: int,
                      dtype: torch.dtype, device: torch.device):
        past_key_values = cls()
        for _i in range(layers):
            k_cache, v_cache = init_kv_cache(
                bsz, n_head, head_dim,
                0, length + cls.KV_ALLOC_BLOCK_LENGTH,
                dtype, device
            )
            past_key_values.key_cache.append(k_cache)
            past_key_values.value_cache.append(v_cache)
        return past_key_values


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


# This function is adapted from
# https://github.com/FasterDecoding/SnapKV/blob/main/snapkv/monkeypatch/snapkv_utils.py
def compress_kv(attn_config, key_states, query_states, value_states, attention_mask,
                num_key_value_groups):
    # check if prefix phase
    invalidInputError(key_states.shape[-2] == query_states.shape[-2], "kv shape mismatch.")
    if not hasattr(attn_config, 'window_size'):
        attn_config.window_size = 32
    if not hasattr(attn_config, 'max_capacity_prompt'):
        attn_config.max_capacity_prompt = 1024
    if not hasattr(attn_config, 'kernel_size'):
        attn_config.kernel_size = 7
    if not hasattr(attn_config, 'pooling'):
        attn_config.pooling = 'maxpool'
    bsz, num_heads, q_len, head_dim = query_states.shape
    if q_len <= attn_config.max_capacity_prompt:
        return key_states, value_states
    else:
        sliding_window_size = getattr(attn_config, "sliding_window", None)
        if sliding_window_size is not None and sliding_window_size <= 2500:
            return key_states[:, :, -sliding_window_size:, :], \
                value_states[:, :, -sliding_window_size:, :]
        else:
            key_states_expand = repeat_kv(key_states, num_key_value_groups).to(key_states.device)
            attn_weights = torch.matmul(query_states[..., -attn_config.window_size:, :],
                                        key_states_expand.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((attn_config.window_size, attn_config.window_size),
                              torch.finfo(attn_weights.dtype).min,
                              device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -attn_config.window_size:,
                         -attn_config.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                                 dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -attn_config.window_size:,
                                            :-attn_config.window_size].sum(dim=-2)
            if attn_config.pooling == 'avgpool':
                if num_key_value_groups > 1:
                    attn_cache = F.avg_pool2d(attn_weights_sum,
                                              kernel_size=(num_key_value_groups,
                                                           attn_config.kernel_size),
                                              padding=(0, attn_config.kernel_size//2),
                                              stride=(num_key_value_groups, 1))
                else:
                    attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=attn_config.kernel_size,
                                              padding=attn_config.kernel_size//2, stride=1)
            elif attn_config.pooling == 'maxpool':
                if num_key_value_groups > 1:
                    attn_cache = F.max_pool2d(attn_weights_sum,
                                              kernel_size=(num_key_value_groups,
                                                           attn_config.kernel_size),
                                              padding=(0, attn_config.kernel_size//2),
                                              stride=(num_key_value_groups, 1))
                else:
                    attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=attn_config.kernel_size,
                                              padding=attn_config.kernel_size//2, stride=1)
            else:
                invalidInputError(False, 'Pooling method not supported')
            indices = attn_cache.topk(attn_config.max_capacity_prompt - attn_config.window_size,
                                      dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-attn_config.window_size, :]\
                .gather(dim=2, index=indices)
            v_past_compress = value_states[:, :, :-attn_config.window_size, :]\
                .gather(dim=2, index=indices)
            k_cur = key_states[:, :, -attn_config.window_size:, :]
            v_cur = value_states[:, :, -attn_config.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            return key_states, value_states


class DynamicCompressCache(DynamicCache):
    def __init__(self, quant_kv=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.real_kv_len = 0

    def update_seen_tokens(self, layer_idx, q_len):
        if layer_idx == 0:
            if hasattr(self, "_seen_tokens"):
                # 4.39 uses `_seen_tokens`
                self._seen_tokens += q_len
            else:
                # 4.37 uses `seen_tokens`
                self.seen_tokens += q_len
            self.real_kv_len += q_len

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        query_states: torch.Tensor,
        attention_mask: torch.Tensor,
        num_key_value_groups: int,
        attn_config: Dict[str, Any],
        enough_kv_room: bool,
        KV_CACHE_ALLOC_BLOCK_LENGTH: int,
        cache_kwargs: Optional[Dict[str, Any]]=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # fix converting empty DynamicCache in transformers >= 4.45
        if key_states == []:
            return key_states, value_states

        bsz, num_heads, seq_len, head_dim = key_states.shape

        if layer_idx == 0:
            if hasattr(self, "_seen_tokens"):
                # 4.39 uses `_seen_tokens`
                self._seen_tokens += seq_len
            else:
                # 4.37 uses `seen_tokens`
                self.seen_tokens += seq_len
            self.real_kv_len += seq_len

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            # First token, compress kv cache
            key_states_compress, value_states_compress = compress_kv(
                attn_config=attn_config,
                key_states=key_states,
                query_states=query_states,
                value_states=value_states,
                attention_mask=attention_mask,
                num_key_value_groups=num_key_value_groups)

            k_cache_compressed, v_cache_compressed = init_kv_cache(
                bsz, num_heads, head_dim,
                0, key_states_compress.size(2) + KV_CACHE_ALLOC_BLOCK_LENGTH,
                key_states.dtype, key_states.device
            )
            k_cache_compressed, v_cache_compressed = append_kv_cache(
                k_cache_compressed, v_cache_compressed,
                key_states_compress, value_states_compress)
            self.key_cache.append(k_cache_compressed)
            self.value_cache.append(v_cache_compressed)

            if key_states.stride(2) != head_dim:
                k_cache, v_cache = init_kv_cache(
                    bsz, num_heads, head_dim,
                    0, key_states.size(2),
                    key_states.dtype, key_states.device
                )
                k_cache, v_cache = append_kv_cache(k_cache, v_cache,
                                                   key_states, value_states)
                return k_cache, v_cache
            else:
                return key_states, value_states
        else:
            cache_k = self.key_cache[layer_idx]
            cache_v = self.value_cache[layer_idx]
            if not enough_kv_room:
                # allocate new
                new_c_k, new_c_v = extend_kv_cache(
                    bsz,
                    num_heads,  # Support GQA
                    head_dim,
                    cache_k.size(2),
                    cache_k.size(2) + KV_CACHE_ALLOC_BLOCK_LENGTH,
                    dtype=cache_k.dtype,
                    device=query_states.device)

                new_c_k[:] = cache_k
                new_c_v[:] = cache_v
                cache_k = new_c_k
                cache_v = new_c_v

            key_states, value_states = append_kv_cache(cache_k,
                                                       cache_v,
                                                       key_states,
                                                       value_states)

            # update past_key_value
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states

            return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer
        index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.real_kv_len

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        num_hidden_layers: int = None
    ) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`. Used for
        backward compatibility."""
        cache = cls(num_hidden_layers)
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                invalidInputError(
                    len(key_states) == 0 and len(value_states) == 0,
                    "from_legacy_cache should be called with an empty kv cache.")
        return cache


class DynamicCompressFp8Cache(DynamicCompressCache, DynamicFp8Cache):
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        query_states: torch.Tensor,
        attention_mask: torch.Tensor,
        num_key_value_groups: int,
        attn_config: Dict[str, Any],
        enough_kv_room: bool,
        KV_CACHE_ALLOC_BLOCK_LENGTH: int,
        cache_kwargs: Optional[Dict[str, Any]]=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # fix converting empty DynamicCache in transformers >= 4.45
        if key_states == []:
            return key_states, value_states

        bsz, num_heads, seq_len, head_dim = key_states.shape

        if layer_idx == 0:
            if hasattr(self, "_seen_tokens"):
                # 4.39 uses `_seen_tokens`
                self._seen_tokens += seq_len
            else:
                # 4.37 uses `seen_tokens`
                self.seen_tokens += seq_len
            self.real_kv_len += seq_len

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            # First token, compress kv cache
            key_states_compress, value_states_compress = compress_kv(
                attn_config=attn_config,
                key_states=key_states,
                query_states=query_states,
                value_states=value_states,
                attention_mask=attention_mask,
                num_key_value_groups=num_key_value_groups)

            k_cache_compressed, v_cache_compressed = init_fp8_kv_cache(
                bsz, num_heads, seq_len, head_dim,
                device=key_states.device,
            )

            k_cache_compressed, v_cache_compressed = append_fp8_kv_cache(
                k_cache_compressed, v_cache_compressed,
                key_states_compress, value_states_compress)
            self.key_cache.append(k_cache_compressed)
            self.value_cache.append(v_cache_compressed)

            if key_states.stride(2) != head_dim:
                k_cache, v_cache = init_fp8_kv_cache(
                    bsz, num_heads, 0, head_dim, key_states.device
                )
                k_cache, v_cache = append_fp8_kv_cache(k_cache, v_cache,
                                                       key_states, value_states)
                return k_cache, v_cache
            else:
                return key_states, value_states
        else:
            cache_k = self.key_cache[layer_idx]
            cache_v = self.value_cache[layer_idx]

            key_states, value_states = append_fp8_kv_cache(cache_k,
                                                           cache_v,
                                                           key_states,
                                                           value_states)

            # update past_key_value
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states

            return self.key_cache[layer_idx], self.value_cache[layer_idx]
