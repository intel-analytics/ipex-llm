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
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from tempfile import NamedTemporaryFile
from .model_implement.baichuan.configuration_baichuan import BaiChuanConfig
from .model_implement.baichuan.modeling_baichuan import BaiChuanForCausalLM
from .model_implement.baichuan.tokenization_baichuan import BaiChuanTokenizer

from ..gguf import GGUFFileLoader


def load_gguf_baichuan(loader: GGUFFileLoader, dtype: torch.dtype = torch.float):
    config = loader.config

    baichuan_config = BaiChuanConfig(
        vocab_size=len(config['tokenizer.ggml.tokens']),
        hidden_size=config['baichuan.embedding_length'],
        intermediate_size=config['baichuan.feed_forward_length'],
        num_hidden_layers=config['baichuan.block_count'],
        num_attention_heads=config['baichuan.attention.head_count'],
        num_key_value_heads=config['baichuan.attention.head_count_kv'],
        hidden_act="silu",
        max_position_embeddings=config['baichuan.context_length'],
        rms_norm_eps=config['baichuan.attention.layer_norm_rms_epsilon'],
        use_cache=True,
        pad_token_id=None,
        bos_token_id=config['tokenizer.ggml.bos_token_id'],
        eos_token_id=config['tokenizer.ggml.eos_token_id'],
        pretraining_tp=1,
    )

    ckpt = loader.tensors(dtype)
    n_head = config['baichuan.attention.head_count']
    n_head_kv = config['baichuan.attention.head_count_kv']
    ckpt = restore_baichuan_weight(ckpt, n_head, n_head_kv)

    state_dict = {}
    state_dict['model.embed_tokens.weight'] = ckpt['token_embd.weight']
    state_dict['model.norm.weight'] = ckpt['output_norm.weight']
    state_dict['lm_head.weight'] = ckpt['output.weight']
    for i in range(config['baichuan.block_count']):
        # rebuild W_pack
        a = ckpt[f'blk.{i}.attn_q.weight']
        b = ckpt[f'blk.{i}.attn_k.weight']
        c = ckpt[f'blk.{i}.attn_v.weight']
        d = torch.cat([a, b, c], dim=0)
        state_dict[f'model.layers.{i}.self_attn.W_pack.weight'] = d

        state_dict[f'model.layers.{i}.self_attn.o_proj.weight'] = \
            ckpt[f'blk.{i}.attn_output.weight']
        state_dict[f'model.layers.{i}.mlp.gate_proj.weight'] = \
            ckpt[f'blk.{i}.ffn_gate.weight']
        state_dict[f'model.layers.{i}.mlp.up_proj.weight'] = \
            ckpt[f'blk.{i}.ffn_up.weight']
        state_dict[f'model.layers.{i}.mlp.down_proj.weight'] = \
            ckpt[f'blk.{i}.ffn_down.weight']
        state_dict[f'model.layers.{i}.input_layernorm.weight'] = \
            ckpt[f'blk.{i}.attn_norm.weight']
        state_dict[f'model.layers.{i}.post_attention_layernorm.weight'] = \
            ckpt[f'blk.{i}.ffn_norm.weight']

    with init_empty_weights():
        model = BaiChuanForCausalLM(baichuan_config)

    for name, weight in state_dict.items():
        set_module_tensor_to_device(model, name, "cpu", weight, dtype=dtype)

    model = model.cpu()

    # see https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto
    from transformers.convert_slow_tokenizer import import_protobuf
    spm_pb2 = import_protobuf("Failed to import protobuf")

    pieces = loader.tokenizer_pieces()
    trainer_spec = spm_pb2.TrainerSpec(byte_fallback=True,
                                       model_type=spm_pb2.TrainerSpec.ModelType.BPE)
    proto = spm_pb2.ModelProto(pieces=pieces, trainer_spec=trainer_spec)
    proto = proto.SerializeToString()

    with NamedTemporaryFile(delete=False) as f:
        f.write(proto)
        f.close()
        tokenizer = BaiChuanTokenizer(f.name)
        os.remove(f.name)

    return model, tokenizer


def restore_baichuan_weight(ckpt: dict, n_head: int, n_head_kv: int):
    # see https://github.com/ggerganov/llama.cpp/blob/master/convert-hf-to-gguf.py#L535

    for name, weight in ckpt.items():
        head, hd_size = weight.shape[0], weight.shape[1:]
        if n_head != n_head_kv:
            new_n_head = n_head // n_head_kv
        else:
            new_n_head = n_head
        if name.endswith("attn_q.weight"):
            ckpt[name] = (weight.reshape(new_n_head, head // new_n_head // 2, 2, *hd_size)
                                .swapaxes(1, 2)
                                .reshape(weight.shape))
        elif name.endswith("attn_k.weight"):
            ckpt[name] = (weight.reshape(new_n_head, head // new_n_head // 2, 2, *hd_size)
                                .swapaxes(1, 2)
                                .reshape(weight.shape))
    return ckpt
