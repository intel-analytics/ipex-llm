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


def convert_lm_head_and_embedding(model, n_splits_linear, temp_dir, weight_dir):
    num_heads = model.model.layers[0].self_attn.num_heads
    head_dim = model.model.layers[0].self_attn.head_dim
    rms_norm_eps = model.config.rms_norm_eps
    vocab_size = model.config.vocab_size
    model_norm = model.model.norm
    lm_head = model.lm_head
    if n_splits_linear == 1:
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
        max_seq_len=1,
        rms_norm_eps=rms_norm_eps,
        mode="decode",
        transpose_value=False,
        dtype=np_dtype,
        model_norm_weight=model_norm.weight.to(torch.float16),
        vocab_size=vocab_size,
        n_splits=n_splits_linear
    )
    last_blob_path = update_names_of_IR_and_export_blob(new_lm_head, "lm_head", temp_dir)

    # save weights bins files
    if n_splits_linear == 1:
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
    )
    first_blob_path = update_names_of_IR_and_export_blob(new_embedding, "embedding",
                                                         temp_dir)
    return first_blob_path, last_blob_path


def convert_baichuan_layer(model, layer_idx, n_splits_linear, n_splits_down_proj,
                           temp_dir, weight_dir, transpose_value_cache, kv_len, group_size,
                           layernorm_const):
    num_heads = model.model.layers[0].self_attn.num_heads
    head_dim = model.model.layers[0].self_attn.head_dim
    intermediate_size = model.config.intermediate_size
    rms_norm_eps = model.config.rms_norm_eps

    from ipex_llm.transformers.npu_models.baichuan_mp import LowBitBaichuanMultiDecoderlayer
    curr_layer = model.model.layers[layer_idx]
    attn_layer = curr_layer.self_attn
    mlp_layer = curr_layer.mlp

    weights = []
    for layer_list in [attn_layer.W_pack_dq_list, attn_layer.o_proj_dq_list,
                       mlp_layer.gate_proj_dq_list, mlp_layer.up_proj_dq_list,
                       mlp_layer.down_proj_dq_list]:
        l_weights = []
        scales = []
        for l in layer_list:
            l_weights.append(l.weight)
            scales.append(l.scale)
        weights.append((torch.stack(l_weights, axis=0), torch.stack(scales, axis=0)))

    cached_cos = curr_layer.self_attn.rotary_emb.cos_cached.to(torch.float16)
    cached_sin = curr_layer.self_attn.rotary_emb.sin_cached.to(torch.float16)
    layer_norm_0 = curr_layer.input_layernorm.weight.to(torch.float16)
    layer_norm_1 = curr_layer.post_attention_layernorm.weight.to(torch.float16)

    if isinstance(weights[0], tuple):
        np_dtype = np.int8 if weights[0][0].dtype == torch.int8 else np.uint8
    else:  # FP16 Linear
        np_dtype = np.float16

    single_decoder = LowBitBaichuanMultiDecoderlayer(
        [1, 1, num_heads * head_dim],
        input_layernorm_weights=[layer_norm_0] if layernorm_const else None,
        post_attn_layernorm_weights=[layer_norm_1] if layernorm_const else None,
        cached_cos=cached_cos,
        cached_sin=cached_sin,
        num_heads=num_heads,
        num_layers=1,
        max_seq_len=kv_len,
        rms_norm_eps=rms_norm_eps,
        intermediate_size=intermediate_size,
        mode="decode",
        transpose_value=transpose_value_cache,
        dtype=np_dtype,
        n_splits_linear=n_splits_linear,
        n_splits_down_proj=n_splits_down_proj,
        group_size=group_size
    )
    rest_blob_path = update_names_of_IR_and_export_blob(single_decoder,
                                                        f"decoder_layer_{layer_idx}",
                                                        temp_dir)

    if layernorm_const:
        st_idx = 5
    else:
        input_lm_bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_3.bin")
        post_lm_bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_4.bin")
        layer_norm_0.data.numpy().tofile(input_lm_bin_file)
        layer_norm_1.data.numpy().tofile(post_lm_bin_file)
        st_idx = 7
    for idx, (weight, scale) in enumerate(weights):
        bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_{st_idx+idx*2}.bin")
        weight.numpy().tofile(bin_file)
        bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_{st_idx+idx*2+1}.bin")
        scale.numpy().tofile(bin_file)
    del single_decoder
