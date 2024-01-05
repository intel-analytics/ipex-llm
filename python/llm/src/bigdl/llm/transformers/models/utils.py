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
from bigdl.llm.utils.common import invalidInputError
from bigdl.llm.transformers.utils import get_ipex_version


def init_kv_cache(batch_size, num_heads, head_dim, current_length, max_length, dtype, device):
    key_cache_storage = torch.empty(batch_size, num_heads,
                                    max_length, head_dim,
                                    dtype=dtype, device=device)
    value_cache_storage = torch.empty(batch_size, num_heads,
                                      max_length, head_dim,
                                      dtype=dtype, device=device)

    key_cache = key_cache_storage.as_strided((batch_size, num_heads,
                                              current_length, head_dim),
                                             key_cache_storage.stride(),
                                             storage_offset=0)
    value_cache = value_cache_storage.as_strided((batch_size, num_heads,
                                                  current_length, head_dim),
                                                 value_cache_storage.stride(),
                                                 storage_offset=0)
    return key_cache, value_cache


def extend_kv_cache(batch_size, num_heads, head_dim, current_length, max_length, dtype, device):
    # empty cache to reduce gpu memory
    if device.type == 'xpu':
        torch.xpu.empty_cache()
    return init_kv_cache(batch_size, num_heads, head_dim, current_length, max_length, dtype, device)


def append_kv_cache(cache_k, cache_v, key_states, value_states):
    new_size = (cache_k.size(0),
                cache_k.size(1),
                cache_k.size(2) + key_states.size(2),
                cache_k.size(3))
    new_cache_k = cache_k.as_strided(new_size, cache_k.stride(), storage_offset=0)
    new_cache_k[:, :, cache_k.size(2):cache_k.size(2) + key_states.size(2), :] = key_states
    new_cache_v = cache_v.as_strided(new_size, cache_v.stride(), storage_offset=0)
    new_cache_v[:, :, cache_v.size(2):cache_k.size(2) + key_states.size(2), :] = value_states
    return new_cache_k, new_cache_v


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, model_family):
    if model_family in ["llama", "baichuan", "internlm", "aquila", "gpt_neox", "mistral",
                        "mixtral"]:
        # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
    elif model_family == "gptj":
        cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        q_embed = (q * cos) + (rotate_every_two(q) * sin)
        k_embed = (k * cos) + (rotate_every_two(k) * sin)
        return q_embed, k_embed
    else:
        invalidInputError(False,
                          f"{model_family} is not supported.")


def apply_rotary_pos_emb_no_cache_xpu(q, k, position_ids, model_family):
    if q.device.type != "xpu":
        invalidInputError(False,
                          f"only xpu is supported in this function")
    import linear_q4_0
    q_embed = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_embed = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    if model_family in ["llama", "baichuan", "internlm", "aquila", "gpt_neox", "mistral",
                        "mixtral"]:
        linear_q4_0.apply_rotary_embedding_half_qk(q, k, position_ids, q_embed, k_embed)
        return q_embed, k_embed
    else:
        invalidInputError(False,
                          f"{model_family} is not supported.")


def is_enough_kv_cache_room_4_36(past_key_value, idx):
    # to determinate if is enough kv cache room in transformers==4.36
    return past_key_value is not None and len(past_key_value.key_cache) > idx and \
        past_key_value.key_cache[idx].stride()[1] > past_key_value.key_cache[idx].size(2) * \
        past_key_value.key_cache[idx].size(3)


def is_enough_kv_cache_room_4_31(past_key_value, seq_len=1):
    # to determinate if is enough kv cache room in transformers between 4.31 and 4.35
    return past_key_value is not None and \
        past_key_value[0].stride()[1] > \
        (past_key_value[0].size(2) + seq_len - 1) * past_key_value[0].size(3)


def use_flash_attention(query, key):
    # here we support query's shape is always [batch_size, head_num, q_len, head_dim],
    # key's shape is always [batch_size, head_num, k_len, head_dim]
    invalidInputError(query.dim() == 4,
                      "Here query input of use_flash_attention should be [batch_size, "
                      "head_num, q_len, head_dim]")
    invalidInputError(key.dim() == 4,
                      "Here key input of use_flash_attention should be [batch_size, "
                      "head_num, k_len, head_dim]")
    bsz, _, q_len, _ = query.size()
    k_len = key.size()[2]
    # check whether ipex flash attention can be used
    if bsz > 1:
        # only use flash attention for batch_size = 1 now
        # as flash attention doesn't support attn_mask in ipex 2.1,
        # so it will cause output error for padded batch input
        return False
    if q_len != k_len:
        # now only use flash attention for first token
        # as it seems have no performance benifit for rest token now
        return False
    if query.device.type != "xpu":
        # ipex flash attention only support for xpu
        return False
    ipex_version = get_ipex_version()
    if ipex_version <= "2.0.110+xpu":
        # ipex flash attention is supported from ipex 2.1
        return False
    if not torch.xpu.has_xetla():
        # ipex flash attention is only supported for xetla
        # may update this later
        return False
    if query.dtype not in [torch.float32, torch.float16]:
        # only use flash attention for fp32/fp16 input
        return False
    return True


def use_esimd_sdp(q_len, head_dim, query_states):
    if head_dim != 128:
        # esimd_sdp only support head_dim = 128 now
        return False
    elif q_len != 1:
        # esimd_sdp only support rest token now
        return False
    elif query_states.device.type != "xpu":
        # esimd_sdp only support GPU now
        return False
    elif query_states.dtype != torch.float16:
        # esimd_sdp only has optimization for FP16 now
        return False
    else:
        device_name = torch.xpu.get_device_name(query_states.device.index)
        if device_name.startswith("Intel(R) Arc(TM) A") or \
                device_name.startswith("Intel(R) Data Center GPU Flex"):
            import linear_fp16_esimd
            if hasattr(linear_fp16_esimd, "sdp_forward"):
                return True
            else:
                return False
        else:
            return False
