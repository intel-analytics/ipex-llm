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
from transformers import FalconConfig, FalconForCausalLM, PreTrainedTokenizerFast

from ..gguf import GGUFFileLoader


def load_gguf_falcon(loader: GGUFFileLoader, dtype: torch.dtype = torch.float):
    config = loader.config

    falcon_config = FalconConfig(
        vocab_size=len(config['tokenizer.ggml.tokens']),
        hidden_size=config['falcon.embedding_length'],
        num_hidden_layers=config['falcon.block_count'],
        num_attention_heads=config['falcon.attention.head_count'],
        num_kv_heads=config['falcon.attention.head_count_kv'],
        max_position_embeddings=config['falcon.context_length'],
        layer_norm_epsilon=config['falcon.attention.layer_norm_epsilon'],
        use_cache=True,
        bos_token_id=config['tokenizer.ggml.bos_token_id'],
        eos_token_id=config['tokenizer.ggml.eos_token_id'],
        # architectures="FalconForCausalLM",
    )

    ckpt = loader.tensors(dtype)
    n_head = config['falcon.attention.head_count']
    n_head_kv = config['falcon.attention.head_count_kv']
    head_dim = config['falcon.embedding_length'] // n_head
    ckpt = restore_falcon_weight(ckpt, n_head, n_head_kv, head_dim)

    state_dict = {}
    state_dict['transformer.word_embeddings.weight'] = ckpt['token_embd.weight']
    state_dict['transformer.ln_f.weight'] = ckpt['output_norm.weight']
    state_dict['transformer.ln_f.bias'] = ckpt['output_norm.bias']
    state_dict['lm_head.weight'] = ckpt['output.weight']
    for i in range(config['falcon.block_count']):
        state_dict[f'transformer.h.{i}.self_attention.query_key_value.weight'] = \
            ckpt[f'blk.{i}.attn_qkv.weight']
        state_dict[f'transformer.h.{i}.self_attention.dense.weight'] = \
            ckpt[f'blk.{i}.attn_output.weight']
        state_dict[f'transformer.h.{i}.mlp.dense_h_to_4h.weight'] = \
            ckpt[f'blk.{i}.ffn_up.weight']
        state_dict[f'transformer.h.{i}.mlp.dense_4h_to_h.weight'] = \
            ckpt[f'blk.{i}.ffn_down.weight']
        state_dict[f'transformer.h.{i}.input_layernorm.weight'] = \
            ckpt[f'blk.{i}.attn_norm.weight']
        state_dict[f'transformer.h.{i}.input_layernorm.bias'] = \
            ckpt[f'blk.{i}.attn_norm.bias']

    with init_empty_weights():
        model = FalconForCausalLM(falcon_config)

    for name, weight in state_dict.items():
        set_module_tensor_to_device(model, name, "cpu", weight, dtype=dtype)

    model = model.cpu()

    pieces, merges = loader.tokenizer_pieces()

    current_directory = os.path.dirname(os.path.abspath(__file__))
    token_file = current_directory + "/model_implement/falcon/tokenizer.json"
    import json
    with open(token_file, 'r') as file:
        data = json.load(file)
    vocab = {}
    # load and replace vocab and merges
    for i in range(len(pieces)):
        token = pieces[i].piece
        score = int(pieces[i].score)
        vocab[token] = score
    data['model']['merges'] = merges
    data['model']['vocab'] = vocab

    with open(token_file, 'w') as file:
        json.dump(data, file, indent=4)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=token_file)
    return model, tokenizer


def restore_falcon_weight(ckpt: dict, n_head: int, n_head_kv: int, head_dim: int):
    # see https://github.com/ggerganov/llama.cpp/blob/
    # master/convert-hf-to-gguf.py#L666
    import numpy as np
    for name, weight in ckpt.items():
        if name.endswith("attn_qkv.weight"):
            part1, part2, part3 = np.split(weight.reshape(-1, head_dim * n_head),
                                           [n_head * head_dim, (n_head + n_head_kv) * head_dim],
                                           axis=0)
            part1 = part1.reshape((n_head_kv, n_head // n_head_kv, head_dim, head_dim * n_head))
            part2 = part2.reshape((n_head_kv, 1, head_dim, head_dim * n_head))
            part3 = part3.reshape((n_head_kv, 1, head_dim, head_dim * n_head))
            data = torch.cat([part1, part2, part3], dim=1)
            ckpt[name] = data.reshape(-1, head_dim * n_head)
    return ckpt
