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
from .common import update_names_of_IR_and_export_blob, obtain_weight_from_single_layer
from intel_npu_acceleration_library.backend.factory import NNFactory
from ipex_llm.transformers.npu_models.mp_models_base import LLMBaseNNFactory
from typing import Sequence


class MiniCPMEmbedding(NNFactory):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        embedding_weight,
        padding_idx,
        dtype,  # fp16
        scale_emb,
        device: str = "NPU",
    ):
        super().__init__(False, device)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.dtype = dtype

        # define input
        weight = self.constant(embedding_weight)
        input = self.parameter((1, 1), dtype=np.int32)

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
        res = res * scale_emb

        # define outputs
        res = self.convert_to_fp16(res)

        print("start compiling")
        self.compile()


class MiniCPMPostEmbedding(NNFactory):
    def __init__(
        self,
        input_size,
        embedding_dim,
        dtype,  # fp16
        scale_emb,
        device: str = "NPU",
    ):
        super().__init__(False, device)
        self.embedding_dim = embedding_dim
        self.dtype = dtype

        input = self.parameter((1, input_size, embedding_dim), dtype=dtype)
        res = input * scale_emb

        # define outputs
        res = self.convert_to_fp16(res)

        print("start compiling")
        self.compile()


class MiniCPMLMHead(LLMBaseNNFactory):
    def __init__(
        self,
        hidden_shape: Sequence[int],
        num_heads: int,
        rms_norm_eps: float,
        model_norm_weight,
        vocab_size: int,
        mode: str = "decode",
        dtype: np.dtype = np.int8,
        max_seq_len: int = 1024,
        transpose_value: bool = False,
        profile: bool = False,
        device: str = "NPU",
        n_splits: int = 1,
        asym: bool = False
    ):
        super().__init__(max_seq_len=max_seq_len,
                         transpose_value=transpose_value,
                         dtype=dtype,
                         profile=profile,
                         device=device)
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.batch_size, self.seq_len, self.hidden_size = hidden_shape
        self.mode = mode
        self.rms_norm_eps = rms_norm_eps
        self.transpose_value = transpose_value
        self.vocab_size = vocab_size

        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads

        # define input, the order self.parameter matters
        input = self.create_input_op((self.batch_size, self.seq_len, self.hidden_size))

        hidden_states = input

        # model norm and lm head
        model_norm_weight = self.constant(model_norm_weight)
        hidden_states = self.layer_norm(hidden_states, model_norm_weight)
        if vocab_size == 122753:
            # for MiniCPM-2B-sft-bf16
            hidden_states_1 = self.linear(
                hidden_states, 73440, self.hidden_size, bias=False, wt_dtype=self.dtype,
                n_splits=n_splits, scale_factor=(n_splits == 1),
                asym=asym
            )
            hidden_states_2 = self.linear(
                hidden_states, 73440, self.hidden_size, bias=False, wt_dtype=self.dtype,
                n_splits=n_splits, scale_factor=(n_splits == 1),
                asym=asym
            )

            hidden_states_2 = self.slice(hidden_states_2, begin=[0, 0, 0], end=[1, 1, 49313])
            hidden_states = self.concat(hidden_states_1, hidden_states_2, axis=2)
        else:
            # for MiniCPM-1B-sft-bf16
            hidden_states = self.linear(
                hidden_states, self.vocab_size, self.hidden_size, bias=False,
                wt_dtype=self.dtype, n_splits=n_splits, scale_factor=(n_splits == 1),
                asym=asym
            )

        # define outputs
        hidden_states = self.convert_to_fp32(hidden_states)

        print("start compiling")
        self.compile()


def convert_lm_head_and_embedding(model, n_splits_linear, temp_dir, weight_dir,
                                  convert_model=False, max_prompt_len=1,
                                  keep_ir=False, compile_blob=True):
    num_heads = model.model.layers[0].self_attn.num_heads
    num_key_value_heads = model.model.layers[0].self_attn.num_key_value_heads
    head_dim = model.model.layers[0].self_attn.head_dim
    rms_norm_eps = model.config.rms_norm_eps
    vocab_size = model.config.vocab_size
    model_norm = model.model.norm
    asym = getattr(model.config, "asym", False)
    if n_splits_linear == 1:
        if vocab_size == 122753:
            # for MiniCPM-2B-sft-bf16
            asym = model.lm_head_0.qtype == "asym_int4_rtn"
            if asym:
                weights = [(model.lm_head_0.weight, model.lm_head_0.scale, model.lm_head_0.zero),
                           (model.lm_head_1.weight, model.lm_head_1.scale, model.lm_head_1.zero)]
            else:
                weights = [(model.lm_head_0.weight, model.lm_head_0.scale),
                           (model.lm_head_1.weight, model.lm_head_1.scale)]
        else:
            # for MiniCPM-1B-sft-bf16
            asym = model.lm_head.qtype == "asym_int4_rtn"
            if asym:
                weights = [(model.lm_head.weight, model.lm_head.scale, model.lm_head.zero)]
            else:
                weights = [(model.lm_head.weight, model.lm_head.scale)]
    else:
        weights = []
        if vocab_size == 122753:
            asym = model.lm_head_0.lm_heads[0].qtype == "asym_int4_rtn"
            lm_head_list = [model.lm_head_0.lm_heads, model.lm_head_1.lm_heads]
        else:
            asym = model.lm_head.lm_heads[0].qtype == "asym_int4_rtn"
            lm_head_list = [model.lm_head.lm_heads]
        for lh in lm_head_list:
            lm_head_weights = []
            scales = []
            zeros = []
            for l in lh:
                lm_head_weights.append(l.weight)
                scales.append(l.scale)
                if l.zero is not None:
                    zeros.append(l.zero)
            if len(zeros):
                weights.append((torch.stack(lm_head_weights, axis=0),
                                torch.stack(scales, axis=0),
                                torch.stack(zeros, axis=0)))
            else:
                weights.append((torch.stack(lm_head_weights, axis=0),
                                torch.stack(scales, axis=0)))
    if isinstance(weights[0], tuple):
        np_dtype = np.int8 if weights[0][0].dtype == torch.int8 else np.uint8
    else:  # FP16 Linear
        np_dtype = np.float16

    new_lm_head = MiniCPMLMHead(
        [1, 1, num_heads * head_dim],
        num_heads=num_heads,
        max_seq_len=1,
        rms_norm_eps=rms_norm_eps,
        mode="decode",
        transpose_value=False,
        dtype=np_dtype,
        model_norm_weight=model_norm.weight.to(torch.float16),
        vocab_size=vocab_size,
        n_splits=n_splits_linear,
        asym=asym
    )
    last_blob_path = update_names_of_IR_and_export_blob(new_lm_head, "lm_head", temp_dir,
                                                        keep_ir=keep_ir, compile_blob=compile_blob)
    os.remove(os.path.join(temp_dir, "lm_head.bin"))

    # save weights bins files
    if n_splits_linear == 1:
        if vocab_size == 122753:
            if not asym:
                weight_numpy = [model.lm_head_0.weight.data.numpy(),
                                model.lm_head_0.scale.data.numpy(),
                                model.lm_head_1.weight.data.numpy(),
                                model.lm_head_1.scale.data.numpy(), ]
            else:
                weight_numpy = [model.lm_head_0.weight.data.numpy(),
                                model.lm_head_0.scale.data.numpy(),
                                model.lm_head_0.zero.data.numpy(),
                                model.lm_head_1.weight.data.numpy(),
                                model.lm_head_1.scale.data.numpy(),
                                model.lm_head_1.zero.data.numpy(), ]
        else:
            if not asym:
                weight_numpy = [model.lm_head.weight.data.numpy(), model.lm_head.scale.data.numpy()]
            else:
                weight_numpy = [model.lm_head.weight.data.numpy(), model.lm_head.scale.data.numpy(),
                                model.lm_head.zero.data.numpy()]
    else:
        weight_numpy = [v.numpy() for v in weights[0]]
        if vocab_size == 122753:
            weight_numpy.extend([v.numpy() for v in weights[1]])

    for idx, weight in enumerate(weight_numpy):
        bin_file = os.path.join(weight_dir, f"model_lm_head_input_{1+idx}.bin")
        weight.tofile(bin_file)

    embedding_layer = model.model.embed_tokens
    new_embedding = MiniCPMEmbedding(
        vocab_size=model.config.vocab_size,
        embedding_dim=model.config.hidden_size,
        embedding_weight=embedding_layer.weight.to(torch.float16).detach().numpy(),
        padding_idx=model.config.pad_token_id,
        dtype=np.float16,
        scale_emb=model.config.scale_emb,
    )
    if convert_model:
        bin_file = os.path.join(weight_dir, f"model_embedding_input_0.bin")
        embedding_layer.weight.to(torch.float16).detach().numpy().tofile(bin_file)
        first_blob_path = None
        # save embedding post module
        embedding_post = MiniCPMPostEmbedding(1, model.config.hidden_size,
                                              dtype=np.float16,
                                              scale_emb=model.config.scale_emb)
        update_names_of_IR_and_export_blob(embedding_post, "embedding_post",
                                           temp_dir, keep_ir=keep_ir, compile_blob=compile_blob)
        embedding_post_prefill = MiniCPMPostEmbedding(max_prompt_len, model.config.hidden_size,
                                                      dtype=np.float16,
                                                      scale_emb=model.config.scale_emb)
        update_names_of_IR_and_export_blob(embedding_post_prefill,
                                           "embedding_post_prefill",
                                           temp_dir, keep_ir=keep_ir, compile_blob=compile_blob)
        os.remove(os.path.join(temp_dir, "embedding_post.bin"))
        os.remove(os.path.join(temp_dir, "embedding_post_prefill.bin"))
    else:
        first_blob_path = update_names_of_IR_and_export_blob(new_embedding, "embedding",
                                                             temp_dir, keep_ir=keep_ir,
                                                             compile_blob=compile_blob)
        os.remove(os.path.join(temp_dir, "embedding.bin"))
    return first_blob_path, last_blob_path


def convert_minicpm_layer(model, layer_idx, n_splits_linear, n_splits_down_proj,
                          temp_dir, weight_dir, transpose_value_cache, kv_len, group_size,
                          const_parameter, mode="decode",
                          keep_ir=False, compile_blob=True):
    num_heads = model.model.layers[0].self_attn.num_heads
    num_key_value_heads = model.model.layers[0].self_attn.num_key_value_heads
    head_dim = model.model.layers[0].self_attn.head_dim
    intermediate_size = model.config.intermediate_size
    rms_norm_eps = model.config.rms_norm_eps
    num_hidden_layers = model.config.num_hidden_layers
    scale_depth = model.model.config.scale_depth
    asym = getattr(model.config, "asym", False)

    from ipex_llm.transformers.npu_models.minicpm_mp import LowBitMinicpmMultiDecoderlayer
    curr_layer = model.model.layers[layer_idx]
    attn_layer = curr_layer.self_attn
    mlp_layer = curr_layer.mlp
    weights = obtain_weight_from_single_layer(attn_layer, mlp_layer)
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
    else:
        input_len = kv_len
        decoder_name = "decoder_layer_prefill"
        const_parameter = False

    single_decoder = LowBitMinicpmMultiDecoderlayer(
        [1, input_len, num_heads * head_dim],
        input_layernorm_weights=[layer_norm_0] if const_parameter else None,
        post_attn_layernorm_weights=[layer_norm_1] if const_parameter else None,
        cached_cos=cached_cos,
        cached_sin=cached_sin,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        num_layers=1,
        max_seq_len=kv_len,
        rms_norm_eps=rms_norm_eps,
        intermediate_size=intermediate_size,
        scale_depth=scale_depth,
        num_hidden_layers=num_hidden_layers,
        mode=mode,
        transpose_value=transpose_value_cache,
        dtype=np_dtype,
        n_splits_linear=n_splits_linear,
        n_splits_down_proj=n_splits_down_proj,
        group_size=group_size,
        asym=asym
    )
    rest_blob_path = update_names_of_IR_and_export_blob(single_decoder,
                                                        decoder_name,
                                                        temp_dir,
                                                        keep_ir=keep_ir, compile_blob=compile_blob)
    os.remove(os.path.join(temp_dir, decoder_name + ".bin"))

    if mode == "decode":
        if const_parameter:
            st_idx = 5
        else:
            input_lm_bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_3.bin")
            post_lm_bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_4.bin")
            layer_norm_0.data.numpy().tofile(input_lm_bin_file)
            layer_norm_1.data.numpy().tofile(post_lm_bin_file)
            st_idx = 7
        if not asym:
            for idx, (weight, scale) in enumerate(weights):
                bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_{st_idx+idx*2}.bin")
                weight.numpy().tofile(bin_file)
                bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_{st_idx+idx*2+1}.bin")
                scale.numpy().tofile(bin_file)
        else:
            for idx, (weight, scale, zero) in enumerate(weights):
                bin_file = os.path.join(weight_dir, f"model_{layer_idx}_input_{st_idx+idx*3}.bin")
                weight.numpy().tofile(bin_file)
                bin_file = os.path.join(weight_dir,
                                        f"model_{layer_idx}_input_{st_idx+idx*3+1}.bin")
                scale.numpy().tofile(bin_file)
                bin_file = os.path.join(weight_dir,
                                        f"model_{layer_idx}_input_{st_idx+idx*3+2}.bin")
                zero.numpy().tofile(bin_file)

        del single_decoder


def convert_fused_minicpm_layer(model, fused_layers, n_splits_linear, n_splits_down_proj,
                                save_dir, weight_dir, transpose_value_cache, kv_len, group_size,
                                const_parameter, mode="decode",
                                keep_ir=False, compile_blob=True):
    num_heads = model.model.layers[0].self_attn.num_heads
    num_key_value_heads = model.model.layers[0].self_attn.num_key_value_heads
    head_dim = model.model.layers[0].self_attn.head_dim
    intermediate_size = model.config.intermediate_size
    rms_norm_eps = model.config.rms_norm_eps
    num_hidden_layers = model.config.num_hidden_layers
    scale_depth = model.model.config.scale_depth
    layer_num = len(model.model.layers)
    fused_layer_num = layer_num // fused_layers
    asym = getattr(model.config, "asym", False)

    from ipex_llm.transformers.npu_models.minicpm_mp import LowBitMinicpmMultiDecoderlayer
    for i in range(fused_layers):
        layer_start = i * fused_layer_num
        layer_end = min((i + 1) * fused_layer_num, layer_num)
        layer_weights = []
        input_layer_norm_weights = []
        post_attn_layernorm_weights = []
        layer_indexs = range(layer_start, layer_end)
        for layer_idx in layer_indexs:
            curr_layer = model.model.layers[layer_idx]
            attn_layer = curr_layer.self_attn
            mlp_layer = curr_layer.mlp
            weights = obtain_weight_from_single_layer(attn_layer, mlp_layer)
            cached_cos = curr_layer.self_attn.rotary_emb.cos_cached.to(torch.float16)
            cached_sin = curr_layer.self_attn.rotary_emb.sin_cached.to(torch.float16)
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
            if not asym:
                for idx, (weight, scale) in enumerate(weights):
                    bin_file = os.path.join(weight_dir,
                                            f"model_{layer_idx}_input_{st_idx+idx*2}.bin")
                    weight.numpy().tofile(bin_file)
                    bin_file = os.path.join(weight_dir,
                                            f"model_{layer_idx}_input_{st_idx+idx*2+1}.bin")
                    scale.numpy().tofile(bin_file)
            else:
                for idx, (weight, scale, zero) in enumerate(weights):
                    bin_file = os.path.join(weight_dir,
                                            f"model_{layer_idx}_input_{st_idx+idx*3}.bin")
                    weight.numpy().tofile(bin_file)
                    bin_file = os.path.join(weight_dir,
                                            f"model_{layer_idx}_input_{st_idx+idx*3+1}.bin")
                    scale.numpy().tofile(bin_file)
                    bin_file = os.path.join(weight_dir,
                                            f"model_{layer_idx}_input_{st_idx+idx*3+2}.bin")
                    zero.numpy().tofile(bin_file)

        if isinstance(weights[0], tuple):
            np_dtype = np.int8 if weights[0][0].dtype == torch.int8 else np.uint8
        else:  # FP16 Linear
            np_dtype = np.float16

        if not const_parameter:
            input_layer_norm_weights = None
            post_attn_layernorm_weights = None

        fused_decoder = LowBitMinicpmMultiDecoderlayer(
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
            scale_depth=scale_depth,
            num_hidden_layers=num_hidden_layers,
            mode=mode,
            transpose_value=transpose_value_cache,
            dtype=np_dtype,
            n_splits_linear=n_splits_linear,
            n_splits_down_proj=n_splits_down_proj,
            group_size=group_size,
            asym=asym
        )
        update_names_of_IR_and_export_blob(fused_decoder,
                                           f"decoder_layer_{i}",
                                           save_dir,
                                           keep_ir=keep_ir, compile_blob=compile_blob)
        os.remove(os.path.join(save_dir, f"decoder_layer_{i}" + ".bin"))
    return 0
