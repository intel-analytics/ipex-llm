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

import os
import torch
from bigdl.llm.utils.common import invalidInputError
from bigdl.llm.ggml.quantize import ggml_tensor_qtype
from bigdl.llm.transformers.utils import get_ipex_version, get_xpu_device_type

FP8_KV_ALLOC_LENGTH = 512
SYM_INT4 = ggml_tensor_qtype["sym_int4"]
SYM_INT8 = ggml_tensor_qtype["sym_int8"]
FP8E4 = ggml_tensor_qtype["fp8_e4m3"]
FP8E5 = ggml_tensor_qtype["fp8_e5m2"]

# used in fused mlp forward
SILU = 0
GELU = 1


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
    new_cache_v[:, :, cache_v.size(2):cache_v.size(2) + key_states.size(2), :] = value_states
    return new_cache_k, new_cache_v


def use_quantize_kv_cache(linear: torch.nn.Module, x: torch.Tensor) -> bool:
    if os.environ.get("BIGDL_QUANTIZE_KV_CACHE", None) is not None:
        return os.environ["BIGDL_QUANTIZE_KV_CACHE"] == "1"
    else:
        return x.device.type == 'xpu' and kv_cache_device_check(x) \
            and hasattr(linear, "qtype") and \
            linear.qtype != ggml_tensor_qtype["fp16"] and linear.qtype != ggml_tensor_qtype["bf16"]


def kv_cache_device_check(x: torch.Tensor) -> bool:
    return get_xpu_device_type(x) == "mtl" or \
        (get_xpu_device_type(x) == "arc" and 1 < x.size(0) and x.size(0) < 8)


def init_fp8_kv_cache(batch_size, num_heads, current_length, head_dim, device):
    max_length = current_length + FP8_KV_ALLOC_LENGTH

    k_cache_storage = torch.empty(batch_size, num_heads, max_length, head_dim,
                                  dtype=torch.uint8, device=device)

    v_cache_storage = torch.empty(batch_size, num_heads, head_dim, max_length,
                                  dtype=torch.uint8, device=device)

    k_cache = k_cache_storage.as_strided((batch_size, num_heads, 0, head_dim),
                                         k_cache_storage.stride(), storage_offset=0)

    v_cache = v_cache_storage.as_strided((batch_size, num_heads, head_dim, 0),
                                         v_cache_storage.stride(), storage_offset=0)

    return k_cache, v_cache.transpose(-1, -2)


def append_fp8_kv_cache(k_cache, v_cache, key, value):
    batch_size, num_heads, cur_length, head_dim = k_cache.shape
    new_length = cur_length + key.size(2)
    new_size = (batch_size, num_heads, new_length, head_dim)

    if k_cache.stride(1) < new_length * k_cache.size(3):
        new_k_cache, new_v_cache = init_fp8_kv_cache(batch_size, num_heads, new_length,
                                                     head_dim, key.device)
        new_k_cache = new_k_cache.as_strided(new_size, new_k_cache.stride(), storage_offset=0)
        new_v_cache = new_v_cache.as_strided(new_size, new_v_cache.stride(), storage_offset=0)
        new_k_cache[:, :, :cur_length, :] = k_cache
        new_v_cache[:, :, :cur_length, :] = v_cache
    else:
        new_k_cache = k_cache.as_strided(new_size, k_cache.stride(), storage_offset=0)
        new_v_cache = v_cache.as_strided(new_size, v_cache.stride(), storage_offset=0)

    fp8_key = key.half().view(torch.uint8)[:, :, :, 1::2]
    new_k_cache[:, :, cur_length:new_length, :] = fp8_key
    fp8_value = value.half().view(torch.uint8)[:, :, :, 1::2]
    new_v_cache[:, :, cur_length:new_length, :] = fp8_value

    return new_k_cache, new_v_cache


def restore_fp8_kv_cache(k_cache, v_cache, dtype):
    new_k_cache = torch.full(k_cache.shape, 128, dtype=torch.int16, device=k_cache.device)
    new_k_cache.view(torch.uint8)[:, :, :, 1::2] = k_cache
    new_k_cache = new_k_cache.view(torch.half)
    new_v_cache = torch.full(v_cache.shape, 128, dtype=torch.int16, device=v_cache.device)
    new_v_cache.view(torch.uint8)[:, :, :, 1::2] = v_cache
    new_v_cache = new_v_cache.view(torch.half)

    return new_k_cache.to(dtype=dtype), new_v_cache.to(dtype=dtype)


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
                        "mixtral", "qwen2", "yuan"]:
        # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
    elif model_family == "gptj":
        q_embed = (q * cos) + (rotate_every_two(q) * sin)
        k_embed = (k * cos) + (rotate_every_two(k) * sin)
        return q_embed, k_embed
    else:
        invalidInputError(False,
                          f"{model_family} is not supported.")


def apply_ipex_rotate_every_two(q, k, cos, sin):
    # ipex's apply_rotary_embedding_two_qk can change the origin storage,
    # so q/k will get the result directly.
    from bigdl.llm.transformers.utils import get_ipex_version
    if get_ipex_version() >= "2.1.10+xpu":
        torch.ops.torch_ipex.apply_rotary_embedding_two_qk(
            q, k, sin, cos, q, k
        )
    else:
        torch.ops.torch_ipex.apply_rotary_embedding(q, sin, cos, q)
        torch.ops.torch_ipex.apply_rotary_embedding(k, sin, cos, k)


def apply_rotary_pos_emb_no_cache_xpu(q, k, position_ids, model_family, rope_theta=10000.0):
    if q.device.type != "xpu":
        invalidInputError(False,
                          f"only xpu is supported in this function")
    import linear_q4_0
    q_embed = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_embed = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    if model_family in ["llama", "baichuan", "internlm", "aquila", "gpt_neox", "mistral",
                        "mixtral"]:
        linear_q4_0.apply_rotary_embedding_half_q_and_k(q, k, position_ids,
                                                        q_embed, k_embed, rope_theta)
        return q_embed, k_embed
    else:
        invalidInputError(False,
                          f"{model_family} is not supported.")


def apply_rotary_pos_emb_cache_freq_xpu(q, k, sin, cos, model_family, position_ids=None):
    if q.device.type != "xpu":
        invalidInputError(False,
                          f"only xpu is supported in this function")
    import linear_q4_0
    q_embed = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_embed = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    if model_family in ["qwen", "mixtral"]:
        linear_q4_0.apply_rotary_embedding_half_q_and_k_cache_freq(q, k, sin, cos, q_embed, k_embed)
    elif model_family in ["qwen2", "yuan"]:
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        linear_q4_0.apply_rotary_embedding_half_q_and_k_cache_freq(q, k, sin, cos, q_embed, k_embed)
    elif model_family in ["gemma"]:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        linear_q4_0.apply_rotary_embedding_half_q_and_k_cache_freq(q, k, sin, cos, q_embed, k_embed)
    else:
        invalidInputError(False,
                          f"{model_family} is not supported.")
    return q_embed, k_embed


def is_enough_kv_cache_room_4_36(past_key_value, idx, seq_len=1):
    # to determinate if is enough kv cache room in transformers==4.36
    # seq_len for current seq len
    # For llama like kv cache, i.e., [bs, n_head, seq_len, head_dim]
    return past_key_value is not None and len(past_key_value.key_cache) > idx and \
        past_key_value.key_cache[idx].stride()[1] >= \
        (past_key_value.key_cache[idx].size(2) + seq_len) * \
        past_key_value.key_cache[idx].size(3)


def is_enough_kv_cache_room_4_31(past_key_value, seq_len=1):
    # to determinate if is enough kv cache room in transformers between 4.31 and 4.35
    # seq_len for current seq len
    # For llama like kv cache, i.e., [bs, n_head, seq_len, head_dim]
    return past_key_value is not None and \
        past_key_value[0].stride()[1] >= \
        (past_key_value[0].size(2) + seq_len) * past_key_value[0].size(3)


def use_flash_attention(query, key, attention_mask=None):
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
    if bsz > 1:
        # as flash attention doesn't support attn_mask in ipex 2.1,
        # so it will cause output error for padded batch input
        if attention_mask is None:
            return True
        else:
            # TODO: below logic may change for different model
            # attention mask shape : [bsz, 1, q_len, k_len]
            if attention_mask[0].squeeze()[0, 0].item() != 0:
                # first batch contains padding
                # otherwise we suppose it should be a upper triangular matrix
                # at the same time, the diagonal is also 0
                return False
            elif not attention_mask.equal(attention_mask[0].repeat(bsz, 1, 1, 1)):
                # check whether mask of every batch is the same
                return False
    return True


def use_esimd_sdp(q_len, k_len, head_dim, query_states):
    if head_dim != 128:
        # esimd_sdp only support head_dim = 128 now
        return False
    elif q_len != 1:
        # esimd_sdp only support rest token now
        return False
    elif k_len < 8:
        # esimd_sdp will cause wrong output when k_len < 8
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


def mlp_fusion_check(x, qtype, training):
    invalidInputError(x.dim() == 2,
                      "Here input x's dim should be 2.")
    if x.shape[0] != 1:
        return False
    if x.device.type != 'xpu':
        return False
    if qtype not in [ggml_tensor_qtype["sym_int4"], ggml_tensor_qtype["fp8_e5m2"],
                     ggml_tensor_qtype["gguf_iq2_xxs"]]:
        return False
    if training or x.requires_grad:
        return False
    return True


def use_xmx(x: torch.Tensor, qtype: int):
    device = get_xpu_device_type(x)
    return (
        os.environ.get("BIGDL_LLM_XMX_DISABLED", "0") != "1"
        and device in ["arc", "flex", "pvc"]
        and qtype in [SYM_INT4, SYM_INT8, FP8E4, FP8E5]
        and (
            (device == "pvc" and 1 < x.size(0) <= 16)
            or
            (device != "pvc" and 1 < x.size(0) <= 64)
        )
    )


def use_fused_layer_norm(x: torch.Tensor, training: bool):
    device = get_xpu_device_type(x)
    return (
        not training
        and not x.requires_grad
        and device in ["arc", "flex", "pvc", "mtl"]  # fused layer norm cannot run on UHD
        and x.numel() // x.size(-1) == 1  # fused layer norm is slower in first token
    )


def fp16_fusion_check(proj, x, training):
    # only use fp16 fusion on PVC inference
    if proj.qtype != ggml_tensor_qtype["fp16"]:
        return False
    if proj.weight_type != 2:
        return False
    if training:
        return False
    if x.requires_grad:
        return False
    device_type = get_xpu_device_type(x)
    if device_type != "pvc":
        return False
    return True
