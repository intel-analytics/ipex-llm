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
from intel_npu_acceleration_library.backend.factory import NNFactory


class Llama32Embedding(NNFactory):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        embedding_weight,
        padding_idx,
        inv_freq,
        attention_scaling,
        dtype,  # fp16
        device: str = "NPU",
    ):
        super().__init__(False, device)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.attention_scaling = attention_scaling
        self.dtype = dtype

        # define input
        weight = self.constant(embedding_weight)
        input = self.parameter((1, 1), dtype=np.int32)
        position_ids = self.parameter((1, 1), dtype=np.int64)
        inv_freq = self.constant(inv_freq)

        # embed_tokens module
        if padding_idx == -1:
            padding_idx += vocab_size

        axis_node = self.constant(np.array([0], dtype=np.int64))
        if padding_idx is not None:
            masked_embeddings = np.ones(weight.shape, dtype=np.float16)
            masked_embeddings[padding_idx, :] = 0.0  # mask

            node_mask = self.constant(masked_embeddings)
            node_masked_w = self.eltwise_mul(weight, node_mask)
            res = self.gather(node_masked_w, input, axis_node, 0)
        else:
            res = self.gather(weight, input, axis_node, 0)

        # rotary_emb module
        inv_freq = self.reshape(inv_freq, (1, inv_freq.shape[0], 1))
        position_ids = self.reshape(position_ids, (1, 1, 1))
        freqs = self.eltwise_mul(self.convert_to_fp32(inv_freq),
                                 self.convert_to_fp32(position_ids))
        freqs = self.transpose(freqs, [0, 2, 1])
        emb = self.concat(freqs, freqs, axis=2)
        cos = self.cos(emb)
        sin = self.sin(emb)
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        # define outputs
        res = self.convert_to_fp16(res)
        cos = self.convert_to_fp32(cos)
        sin = self.convert_to_fp32(sin)

        print("start compiling")
        self.compile()


class Llama32PostEmbedding(NNFactory):
    def __init__(
        self,
        inv_freq,
        attention_scaling,
        input_len: int = 1,
        device: str = "NPU",
    ):
        super().__init__(False, device)
        self.attention_scaling = attention_scaling

        # define input
        position_ids = self.parameter((1, input_len), dtype=np.int64)
        inv_freq = self.constant(inv_freq)

        # rotary_emb module
        inv_freq = self.reshape(inv_freq, (1, inv_freq.shape[0], 1))
        position_ids = self.reshape(position_ids, (1, 1, input_len))
        freqs = self.eltwise_mul(self.convert_to_fp32(inv_freq),
                                 self.convert_to_fp32(position_ids))
        freqs = self.transpose(freqs, [0, 2, 1])
        emb = self.concat(freqs, freqs, axis=2)
        cos = self.cos(emb)
        sin = self.sin(emb)
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        if input_len > 1:
            cos = self.unsqueeze(cos, [1])
            sin = self.unsqueeze(sin, [1])

        # define outputs
        cos = self.convert_to_fp32(cos)
        sin = self.convert_to_fp32(sin)

        print("start compiling")
        self.compile()


def convert_lm_head_and_embedding(model, n_splits_linear, temp_dir, weight_dir,
                                  convert_model=False, max_prompt_len=1):
    num_heads = model.model.layers[0].self_attn.num_heads
    num_key_value_heads = model.model.layers[0].self_attn.num_key_value_heads
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
        for i in range(n_splits_linear):
            lm_head_weights.append(lm_heads[i].weight)
            scales.append(lm_heads[i].scale)
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
    last_blob_path = update_names_of_IR_and_export_blob(new_lm_head, "lm_head", temp_dir,
                                                        True, False)

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

    if hasattr(model.model.layers[0].self_attn.rotary_emb, "cos_cached"):
        # llama-2-7B & llama-3-8B
        embedding_layer = model.model.embed_tokens
        new_embedding = LLMEmbedding(
            vocab_size=model.config.vocab_size,
            embedding_dim=model.config.hidden_size,
            embedding_weight=embedding_layer.weight.to(torch.float16).detach().numpy(),
            padding_idx=model.config.pad_token_id,
            dtype=np.float16,
        )
        if convert_model:
            bin_file = os.path.join(weight_dir, f"model_embedding_input_0.bin")
            embedding_layer.weight.to(torch.float16).detach().numpy().tofile(bin_file)
            first_blob_path = None
        else:
            first_blob_path = update_names_of_IR_and_export_blob(new_embedding, "embedding",
                                                                 temp_dir, True, False)
    else:
        # llama-3.2-3B & llama-3.2-1B
        embedding_layer = model.model.embed_tokens
        new_embedding = Llama32Embedding(
            vocab_size=model.config.vocab_size,
            embedding_dim=model.config.hidden_size,
            embedding_weight=embedding_layer.weight.to(torch.float16).detach().numpy(),
            padding_idx=model.config.pad_token_id,
            inv_freq=model.model.rotary_emb.inv_freq.to(torch.float16),
            attention_scaling=model.model.rotary_emb.attention_scaling,
            dtype=np.float16,
        )
        if convert_model:
            bin_file = os.path.join(weight_dir, f"model_embedding_input_0.bin")
            embedding_layer.weight.to(torch.float16).detach().numpy().tofile(bin_file)
            first_blob_path = None
            # save embedding post module
            inv_freq = model.model.rotary_emb.inv_freq.to(torch.float16)
            attention_scaling = model.model.rotary_emb.attention_scaling
            embedding_post = Llama32PostEmbedding(inv_freq=inv_freq,
                                                  attention_scaling=attention_scaling,
                                                  input_len=1)
            update_names_of_IR_and_export_blob(embedding_post, "embedding_post",
                                               temp_dir, True, False)
            embedding_post_prefill = Llama32PostEmbedding(inv_freq=inv_freq,
                                                          attention_scaling=attention_scaling,
                                                          input_len=max_prompt_len)
            update_names_of_IR_and_export_blob(embedding_post_prefill,
                                               "embedding_post_prefill",
                                               temp_dir, True, False)
        else:
            first_blob_path = update_names_of_IR_and_export_blob(new_embedding, "embedding",
                                                                 temp_dir)
    return first_blob_path, last_blob_path


def convert_llama_layer(model, layer_idx, n_splits_linear, n_splits_down_proj,
                        temp_dir, weight_dir, transpose_value_cache, kv_len, group_size,
                        layernorm_const, mode="decode"):
    num_heads = model.model.layers[0].self_attn.num_heads
    num_key_value_heads = model.model.layers[0].self_attn.num_key_value_heads
    head_dim = model.model.layers[0].self_attn.head_dim
    intermediate_size = model.config.intermediate_size
    rms_norm_eps = model.config.rms_norm_eps

    from ipex_llm.transformers.npu_models.llama_mp import LowBitLlamaMultiDecoderlayer
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

    if hasattr(curr_layer.self_attn.rotary_emb, "cos_cached"):
        # llama-2-7B & llama-3-8B
        cached_cos = curr_layer.self_attn.rotary_emb.cos_cached.to(torch.float16)
        cached_sin = curr_layer.self_attn.rotary_emb.sin_cached.to(torch.float16)
    else:
        # llama-3.2-3B & llama-3.2-1B
        cached_cos = None
        cached_sin = None
    layer_norm_0 = curr_layer.input_layernorm.weight.to(torch.float16)
    layer_norm_1 = curr_layer.post_attention_layernorm.weight.to(torch.float16)

    if isinstance(weights[0], tuple):
        np_dtype = np.int8 if weights[0][0].dtype == torch.int8 else np.uint8
    else:  # FP16 Linear
        np_dtype = np.float16

    if mode == "decode":
        input_len = 1
        decoder_name = f"decoder_layer_{layer_idx}"
        keep_position_ids = True
        npu_dpu_groups = None
    else:
        input_len = kv_len
        decoder_name = "decoder_layer_prefill"
        layernorm_const = False
        keep_position_ids = False
        npu_dpu_groups = 6

    single_decoder = LowBitLlamaMultiDecoderlayer(
        [1, input_len, num_heads * head_dim],
        input_layernorm_weights=[layer_norm_0] if layernorm_const else None,
        post_attn_layernorm_weights=[layer_norm_1] if layernorm_const else None,
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
        group_size=group_size,
        cos_len=input_len,
        keep_position_ids=keep_position_ids
    )

    rest_blob_path = update_names_of_IR_and_export_blob(single_decoder,
                                                        decoder_name,
                                                        temp_dir,
                                                        True, False,
                                                        npu_dpu_groups=npu_dpu_groups)

    if mode == "decode":
        if hasattr(curr_layer.self_attn.rotary_emb, "cos_cached"):
            # llama-2-7B & llama-3-8B
            if layernorm_const:
                st_idx = 5
            else:
                input_lm_bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_3.bin")
                post_lm_bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_4.bin")
                layer_norm_0.data.numpy().tofile(input_lm_bin_file)
                layer_norm_1.data.numpy().tofile(post_lm_bin_file)
                st_idx = 7
        else:
            # llama-3.2-3B & llama-3.2-1B
            if layernorm_const:
                st_idx = 6
            else:
                input_lm_bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_4.bin")
                post_lm_bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_5.bin")
                layer_norm_0.data.numpy().tofile(input_lm_bin_file)
                layer_norm_1.data.numpy().tofile(post_lm_bin_file)
                st_idx = 8
        for idx, (weight, scale) in enumerate(weights):
            bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_{st_idx+idx*2}.bin")
            weight.numpy().tofile(bin_file)
            bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_{st_idx+idx*2+1}.bin")
            scale.numpy().tofile(bin_file)
        del single_decoder


def convert_fused_llama_layer(model, fused_layers, n_splits_linear, n_splits_down_proj,
                              save_dir, weight_dir, transpose_value_cache, kv_len, group_size,
                              layernorm_const, mode="decode"):
    num_heads = model.model.layers[0].self_attn.num_heads
    num_key_value_heads = model.model.layers[0].self_attn.num_key_value_heads
    head_dim = model.model.layers[0].self_attn.head_dim
    intermediate_size = model.config.intermediate_size
    rms_norm_eps = model.config.rms_norm_eps
    layer_num = len(model.model.layers)
    fused_layer_num = layer_num // fused_layers

    from ipex_llm.transformers.npu_models.llama_mp import LowBitLlamaMultiDecoderlayer
    for i in range(fused_layers):
        layer_start = i * fused_layer_num
        layer_end = min((i + 1) * fused_layer_num, layer_num)
        layer_weights = []
        input_layer_norm_weights = []
        post_attn_layernorm_weights = []
        layer_indexs = range(layer_start, layer_end)
        n_splits_linear = len(model.model.layers[0].mlp.gate_proj_dq_list)
        n_splits_down_proj = len(model.model.layers[0].mlp.down_proj_dq_list)
        for layer_idx in layer_indexs:
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

            if hasattr(curr_layer.self_attn.rotary_emb, "cos_cached"):
                # llama-2-7B & llama-3-8B
                cached_cos = curr_layer.self_attn.rotary_emb.cos_cached.to(torch.float16)
                cached_sin = curr_layer.self_attn.rotary_emb.sin_cached.to(torch.float16)
            else:
                # llama-3.2-3B & llama-3.2-1B
                cached_cos = None
                cached_sin = None
            layer_norm_0 = curr_layer.input_layernorm.weight.to(torch.float16)
            layer_norm_1 = curr_layer.post_attention_layernorm.weight.to(torch.float16)

            layer_weights.extend(weights)
            input_layer_norm_weights.append(layer_norm_0)
            post_attn_layernorm_weights.append(layer_norm_1)

            # save weight
            input_lm_bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_3.bin")
            post_lm_bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_4.bin")
            layer_norm_0.data.numpy().tofile(input_lm_bin_file)
            layer_norm_1.data.numpy().tofile(post_lm_bin_file)
            st_idx = 5
            # 6, 7 are past k/v
            for idx, (weight, scale) in enumerate(weights):
                bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_{st_idx+idx*2}.bin")
                weight.numpy().tofile(bin_file)
                bin_file = os.path.join(weight_dir,
                                        f"model_{layer_idx}_input_{st_idx+idx*2+1}.bin")
                scale.numpy().tofile(bin_file)

        if isinstance(weights[0], tuple):
            np_dtype = np.int8 if weights[0][0].dtype == torch.int8 else np.uint8
        else:  # FP16 Linear
            np_dtype = np.float16

        fused_decoder = LowBitLlamaMultiDecoderlayer(
            [1, 1, num_heads * head_dim],
            input_layernorm_weights=input_layer_norm_weights,
            post_attn_layernorm_weights=post_attn_layernorm_weights,
            cached_cos=cached_cos,
            cached_sin=cached_sin,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            num_layers=fused_layer_num,
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
        update_names_of_IR_and_export_blob(fused_decoder,
                                           f"decoder_layer_{i}",
                                           save_dir,
                                           compile_blob=True,
                                           keep_ir=False)
    return 0
