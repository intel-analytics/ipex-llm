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

from typing import List, Optional, Tuple, Union
import warnings
import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import time
import argparse

from ipex_llm.transformers import AutoModelForCausalLM
from ipex_llm.transformers.convert import convert_forward
from ipex_llm.transformers.models.common import scaled_dot_product_attention
from ipex_llm.transformers.models.common import rms_norm_forward
from ipex_llm.transformers.models.common import mlp_silu_forward
from ipex_llm.transformers.kv import DynamicNormalCache
from ipex_llm.utils.benchmark_util_deepseek import BenchmarkWrapper

from transformers import AutoTokenizer, GenerationConfig
from transformers.cache_utils import Cache, DynamicCache
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_13
from transformers.utils.import_utils import is_torch_fx_available


PROMPT_FORMAT = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {prompt}.
Assistant: <think>
"""


def convert_forward_to_xpu(m, target_m, new_forward):
    # print(m.__class__.__name__)
    if m.__class__.__name__ == target_m:
        bound_method = new_forward.__get__(m, m.__class__)
        setattr(m, "forward", bound_method)
        # m = m.to(device="xpu", dtype=torch.float16)
    for _, sub_m in m.named_children():
        convert_forward_to_xpu(sub_m, target_m, new_forward)


def hybrid_DeepseekV3MoE_forward(self, hidden_states):
    # convert1_start = time.time()
    hidden_states = hidden_states.to(device="cpu")#, dtype=torch.bfloat16)
    # convert1_end = time.time()
    # moe_start = time.time()
    identity = hidden_states
    orig_shape = hidden_states.shape
    topk_idx, topk_weight = self.gate(hidden_states)
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    if not self.training:
        y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
    if self.config.n_shared_experts is not None:
        y = y + self.shared_experts(identity)
    # moe_end = time.time()
    # convert2_start = time.time()
    y = y.to(device="xpu")#, dtype=torch.float16)
    # convert2_end = time.time()
    # print("convert to cpu time: ", (convert1_end - convert1_start)*1000)
    # print("moe time: ", (moe_end - moe_start) * 1000)
    # print("convert to xpu time: ", (convert2_end - convert2_start) * 1000)
    return y


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def hybrid_DeepseekV3Attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )
    bsz, q_len, _ = hidden_states.size()

    if self.q_lora_rank is None:
        q = self.q_proj(hidden_states)
    else:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(
        q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    compressed_kv, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    kv = (
        self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        .transpose(1, 2)
    )

    k_nope, value_states = torch.split(
        kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
    )
    kv_seq_len = value_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

    query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

    key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        padded_value_states = torch.zeros([value_states.shape[0], value_states.shape[1], value_states.shape[2], key_states.shape[-1]],
                                          dtype=value_states.dtype, device=value_states.device)
        padded_value_states[:, :, :, :value_states.shape[-1]] = value_states
        key_states, value_states = past_key_value.update(
            key_states, padded_value_states, self.layer_idx, cache_kwargs
        )

    attn_weights = None
    # import pdb
    # breakpoint()
    attn_output = scaled_dot_product_attention(
        query_states, key_states, value_states,
        attention_mask, q_len == kv_seq_len, scale=self.softmax_scale
    )
    attn_output = attn_output[:, :, :, :self.v_head_dim]

    # attn_weights = (
    #     torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
    # )
    #
    # if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
    #     raise ValueError(
    #         f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
    #         f" {attn_weights.size()}"
    #     )
    # assert attention_mask is not None
    # if attention_mask is not None:
    #     if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
    #         raise ValueError(
    #             f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
    #         )
    #     attn_weights = attn_weights + attention_mask
    #
    # # upcast attention to fp32
    # attn_weights = nn.functional.softmax(
    #     attn_weights, dim=-1, dtype=torch.float32
    # ).to(query_states.dtype)
    # attn_weights = nn.functional.dropout(
    #     attn_weights, p=self.attention_dropout, training=self.training
    # )
    # attn_output = torch.matmul(attn_weights, value_states)

    # if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
    #     raise ValueError(
    #         f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
    #         f" {attn_output.size()}"
    #     )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def do_benchmark(layer, num_warmup=3, num_trials=10, device="xpu", **kwargs):
    for i in range(num_warmup):
        layer(**kwargs)
        if device == "xpu":
            torch.xpu.synchronize()

    total_time = 0
    for i in range(num_trials):
        start_time = time.time()
        output = layer(**kwargs)
        if device == "xpu":
            torch.xpu.synchronize()
        end_time = time.time()
        total_time += (end_time - start_time)
    average = total_time * 1000 / num_trials
    print("{} latency: {} ms".format(layer.__class__.__name__, average))
    if device == "xpu":  # TODO: need to empty cache after each run?
        torch.xpu.empty_cache()
    return average


# kvcache will increment after each run, can't reuse the same input to run multiple trials
def do_benchmark_attn(layer, hidden_states, num_warmup=3, num_trials=128, device="xpu"):
    kv_seq_length = 128 - num_warmup  # Simulate the average of 128-128
    past_key = torch.randn(1, 128, kv_seq_length, 192, dtype=hidden_states.dtype).to(device)
    past_value = torch.randn(1, 128, kv_seq_length, 128, dtype=hidden_states.dtype).to(device)  # Not padded
    # past_key_values = DynamicNormalCache()
    # past_key_values.update(past_key, past_value, 0)
    # past_key_values.update(past_key, past_value, 1)
    past_key_values = DynamicCache.from_legacy_cache([(past_key, past_value), (past_key, past_value)])  # kv for 2 layers
    total_time = 0
    for i in range(num_warmup+num_trials):
        position_ids = torch.tensor([[kv_seq_length]]).to(device)
        attention_mask = torch.zeros([1, 1, 1, kv_seq_length + 1], dtype=hidden_states.dtype).to(device)
        start_time = time.time()
        output = layer(hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                       past_key_value=past_key_values, output_attention=False, use_cache=True)
        if device == "xpu":
            torch.xpu.synchronize()
        end_time = time.time()
        kv_seq_length += 1
        if i >= num_warmup:
            total_time += (end_time-start_time)
            # print((end_time-start_time)*1000)
    average = total_time * 1000 / num_trials
    print("{} latency: {} ms".format(layer.__class__.__name__, average))
    if device == "xpu":
        torch.xpu.empty_cache()
    return average


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--load-path', type=str, default=None,
                        help='The path to load the low-bit model.')
    parser.add_argument('--warm-up', type=int, default=3,
                        help='Num of warm-up trials.')
    parser.add_argument('--num-trials', type=int, default=128,
                        help='Num of trials to run.')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    load_path = args.load_path
    if load_path:
        model = AutoModelForCausalLM.load_low_bit(load_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(load_path,
                                              trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                    load_in_4bit=True,
                                                    optimize_model=True,
                                                    trust_remote_code=True,
                                                    use_cache=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)

    print(model)

    # device = "cpu"
    device = "xpu"
    input_ids = torch.tensor([[1128]]).to(device)
    hidden_states = torch.randn(1, 1, 7168)
    if device == "xpu":
        convert_forward_to_xpu(model.model, "DeepseekV3MoE", hybrid_DeepseekV3MoE_forward)
        # convert_forward_to_xpu(model.model, "DeepseekV3Attention", hybrid_DeepseekV3Attention_forward)
        for i in range(0, model.config.num_hidden_layers):
            model.model.layers[i].input_layernorm = model.model.layers[i].input_layernorm.to(device="xpu")#, dtype=torch.float16)
            model.model.layers[i].self_attn = model.model.layers[i].self_attn.to(device="xpu")#, dtype=torch.float16)
            model.model.layers[i].post_attention_layernorm = model.model.layers[i].post_attention_layernorm.to(device="xpu")#, dtype=torch.float16)
            if i < model.config.first_k_dense_replace:
                model.model.layers[i].mlp = model.model.layers[i].mlp.to(device="xpu")#, dtype=torch.float16)
            # else:
                # model.model.layers[i].mlp.gate = model.model.layers[i].mlp.gate.to(device="xpu", dtype=torch.float16)
                # model.model.layers[i].mlp.shared_experts = model.model.layers[i].mlp.shared_experts.to(device="xpu", dtype=torch.float16)
        model.model.embed_tokens = model.model.embed_tokens.to(device="xpu")#, dtype=torch.float16)
        model.model.norm = model.model.norm.to(device="xpu")#, dtype=torch.float16)
        model.lm_head = model.lm_head.to(device="xpu")#, dtype=torch.float16)
        # hidden_states = hidden_states.half()
        convert_forward_to_xpu(model, "DeepseekV3RMSNorm", rms_norm_forward)
        convert_forward_to_xpu(model, "DeepseekV3MLP", mlp_silu_forward)
    else:  # cpu, bf16
        model = model.bfloat16()
        hidden_states = hidden_states.bfloat16()
    hidden_states = hidden_states.to(device)

    # Breakdown of e2e
    embed = model.model.embed_tokens
    embed_time = do_benchmark(embed, args.warm_up, args.num_trials, device, input=input_ids)

    dense_decoder = model.model.layers[0]
    dense_decoder_time = do_benchmark_attn(dense_decoder, hidden_states, args.warm_up, num_trials=128, device=device)

    moe_decoder = model.model.layers[1]
    moe_decoder_time = do_benchmark_attn(moe_decoder, hidden_states, args.warm_up, num_trials=128, device=device)

    norm = model.model.norm
    norm_time = do_benchmark(norm, args.warm_up, args.num_trials, device, hidden_states=hidden_states)

    lm_head = model.lm_head
    lm_head_time = do_benchmark(lm_head, args.warm_up, args.num_trials, device, x=hidden_states)

    total_time = embed_time + dense_decoder_time + moe_decoder_time +norm_time + lm_head_time
    print("Overall latency: {} ms".format(total_time))
    print("==================")

    # Breakdown of decoder layer
    self_attn = model.model.layers[0].self_attn
    do_benchmark_attn(self_attn, hidden_states, args.warm_up, num_trials=128, device=device)

    mlp = model.model.layers[0].mlp
    do_benchmark(mlp, args.warm_up, args.num_trials, device, x=hidden_states)

    input_norm = model.model.layers[0].input_layernorm
    do_benchmark(input_norm, args.warm_up, args.num_trials, device, hidden_states=hidden_states)

    post_norm = model.model.layers[0].post_attention_layernorm
    do_benchmark(post_norm, args.warm_up, args.num_trials, device, hidden_states=hidden_states)

    moe = model.model.layers[1].mlp  # including cpu/xpu data conversion
    do_benchmark(moe, args.warm_up, args.num_trials, device, hidden_states=hidden_states)
