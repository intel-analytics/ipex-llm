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
import numpy as np
import os
from .common import update_names_of_IR_and_export_blob, LLMEmbedding, LowBitLLMLMHead
from ipex_llm.transformers.npu_models.lm_head import SlicedLMHead


def convert_lm_head_and_embedding(model, n_splits_linear, temp_dir, weight_dir,
                                  compile_full_model=False):
    num_heads = model.model.layers[0].self_attn.num_heads
    head_dim = model.model.layers[0].self_attn.head_dim
    rms_norm_eps = model.config.rms_norm_eps
    vocab_size = model.config.vocab_size
    model_norm = model.model.norm
    lm_head = model.lm_head
    if not isinstance(lm_head, SlicedLMHead):
        weights = [(lm_head.weight, lm_head.scale)]
    else:
        lm_heads = lm_head.lm_heads
        lm_head_weights = []
        scales = []
        for l in lm_heads:
            lm_head_weights.append(l.weight)
            scales.append(l.scale)
        weights = [(torch.stack(lm_head_weights, axis=0),
                    torch.stack(scales, axis=0))]
    if isinstance(weights[0], tuple):
        np_dtype = np.int8 if weights[0][0].dtype == torch.int8 else np.uint8
    else:  # FP16 Linear
        np_dtype = np.float16

    new_lm_head = LowBitLLMLMHead(
        [1, 1, num_heads * head_dim],
        num_heads=num_heads,
        max_seq_len=1,  # seems doesn't matter
        rms_norm_eps=rms_norm_eps,
        mode="decode",
        transpose_value=False,  # seems doesn't matter
        dtype=np_dtype,
        model_norm_weight=model_norm.weight.to(torch.float16),
        vocab_size=vocab_size,
        n_splits=n_splits_linear
    )

    last_blob_path = update_names_of_IR_and_export_blob(new_lm_head, f"lm_head",
                                                        temp_dir, True, True)

    # save weights bins files
    if not isinstance(lm_head, SlicedLMHead):
        weight_numpy = [
            lm_head.weight.data.numpy(), lm_head.scale.data.numpy(),
        ]
    else:
        weight_numpy = [v.numpy() for v in weights[0]]

    for idx, weight in enumerate(weight_numpy):
        bin_file = os.path.join(weight_dir, f"model_lm_head_input_{1+idx}.bin")
        weight.tofile(bin_file)

    embedding_layer = model.model.embed_tokens
    new_embedding = LLMEmbedding(
        vocab_size=model.config.vocab_size,
        embedding_dim=model.config.hidden_size,
        embedding_weight=embedding_layer.weight.to(torch.float16).detach().numpy(),
        padding_idx=model.config.pad_token_id,
        dtype=np.float16,
        input_length=1,
    )
    first_blob_path = update_names_of_IR_and_export_blob(new_embedding, f"embedding",
                                                         temp_dir, True, keep_ir=True)
    if compile_full_model:
        bin_file = os.path.join(weight_dir, f"model_embedding_input_0.bin")
        embedding_layer.weight.to(torch.float16).detach().numpy().tofile(bin_file)
    return first_blob_path, last_blob_path


def convert_qwen_layer(model, layer_idx, n_splits_linear, n_splits_down_proj,
                       temp_dir, weight_dir, transpose_value_cache, kv_len, group_size,
                       layernorm_const, mode="decode"):
    num_heads = model.model.layers[0].self_attn.num_heads
    num_key_value_heads = model.model.layers[0].self_attn.num_key_value_heads
    head_dim = model.model.layers[0].self_attn.head_dim
    intermediate_size = model.config.intermediate_size
    rms_norm_eps = model.config.rms_norm_eps

    from ipex_llm.transformers.npu_models.qwen2_mp import LowBitQwenMultiDecoderlayer
    curr_layer = model.model.layers[layer_idx]
    attn_layer = curr_layer.self_attn
    mlp_layer = curr_layer.mlp

    weights = []
    for layer_list in [attn_layer.q_proj_dq_list, attn_layer.k_proj_dq_list,
                       attn_layer.v_proj_dq_list, attn_layer.o_proj_dq_list,
                       mlp_layer.gate_proj_dq_list, mlp_layer.up_proj_dq_list,
                       mlp_layer.down_proj_dq_list]:
        l_weights = []
        scales = []
        for l in layer_list:
            l_weights.append(l.weight)
            scales.append(l.scale)
        weights.append((torch.stack(l_weights, axis=0), torch.stack(scales, axis=0)))

    q_bias = attn_layer.q_proj_dq_list.q_proj_dq_0.bias.to(torch.float16)
    k_bias = attn_layer.k_proj_dq_list.k_proj_dq_0.bias.to(torch.float16)
    v_bias = attn_layer.v_proj_dq_list.v_proj_dq_0.bias.to(torch.float16)
    cached_cos = curr_layer.self_attn.rotary_emb.cos_cached.to(torch.float16)
    cached_sin = curr_layer.self_attn.rotary_emb.sin_cached.to(torch.float16)
    layer_norm_0 = curr_layer.input_layernorm.weight.to(torch.float16)
    layer_norm_1 = curr_layer.post_attention_layernorm.weight.to(torch.float16)

    if isinstance(weights[0], tuple):
        np_dtype = np.int8 if weights[0][0].dtype == torch.int8 else np.uint8
    else:  # FP16 Linear
        np_dtype = np.float16

    if mode == "decode":
        input_len = 1
        decoder_name = f"decoder_layer_{layer_idx}"
        compile = True
        keep_ir = True
    else:
        input_len = kv_len
        decoder_name = "decoder_layer_prefill"
        compile = False
        keep_ir = True
    single_decoder = LowBitQwenMultiDecoderlayer(
        [1, input_len, num_heads * head_dim],
        input_layernorm_weights=None,
        post_attn_layernorm_weights=None,
        q_biases=None,
        k_biases=None,
        v_biases=None,
        cached_cos=cached_cos,
        cached_sin=cached_sin,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        num_layers=1,
        max_seq_len=kv_len,
        rms_norm_eps=rms_norm_eps,
        intermediate_size=intermediate_size,
        mode=mode,
        transpose_value=transpose_value_cache,
        dtype=np_dtype,
        n_splits_linear=n_splits_linear,
        n_splits_down_proj=n_splits_down_proj,
        group_size=group_size
    )
    rest_blob_path = update_names_of_IR_and_export_blob(single_decoder,
                                                        decoder_name,
                                                        temp_dir, compile, keep_ir)

    # 0, 1, 2 are input_embed/attention_mask/position_id
    if mode == "decode":
        if layernorm_const:
            st_idx = 3
        else:
            input_lm_bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_3.bin")
            post_lm_bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_4.bin")
            layer_norm_0.data.numpy().tofile(input_lm_bin_file)
            layer_norm_1.data.numpy().tofile(post_lm_bin_file)
            st_idx = 5
        q_bias_bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_{st_idx}.bin")
        k_bias_bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_{st_idx+1}.bin")
        v_bias_bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_{st_idx+2}.bin")
        q_bias.data.numpy().tofile(q_bias_bin_file)
        k_bias.data.numpy().tofile(k_bias_bin_file)
        v_bias.data.numpy().tofile(v_bias_bin_file)
        # 6, 7 are past k/v
        for idx, (weight, scale) in enumerate(weights):
            bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_{st_idx+5+idx*2}.bin")
            weight.numpy().tofile(bin_file)
            bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_{st_idx+5+idx*2+1}.bin")
            scale.numpy().tofile(bin_file)

    del single_decoder
