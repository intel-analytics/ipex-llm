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
import os
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import BloomConfig, BloomForCausalLM, BloomTokenizerFast

from ..gguf import GGUFFileLoader


def load_gguf_bloom(loader: GGUFFileLoader, dtype: torch.dtype = torch.float):
    config = loader.config

    bloom_config = BloomConfig(
        vocab_size=len(config['tokenizer.ggml.tokens']),
        hidden_size=config['bloom.embedding_length'],
        n_layer=config['bloom.block_count'],
        n_head=config['bloom.attention.head_count'],
        rms_norm_eps=config['bloom.attention.layer_norm_epsilon'],
        use_cache=True,
        pad_token_id=config['tokenizer.ggml.padding_token_id'],
        unknown_token_id=config['tokenizer.ggml.unknown_token_id'],
        bos_token_id=config['tokenizer.ggml.bos_token_id'],
        eos_token_id=config['tokenizer.ggml.eos_token_id'],
        pretraining_tp=1,
    )

    ckpt = loader.tensors(dtype)
    n_head = config['bloom.attention.head_count']
    n_embed = config['bloom.embedding_length']
    ckpt = restore_bloom_weight(ckpt, n_head, n_embed)

    state_dict = {}
    state_dict['transformer.word_embeddings.weight'] = ckpt['token_embd.weight']
    state_dict['transformer.word_embeddings_layernorm.weight'] = ckpt['token_embd_norm.weight']
    state_dict['transformer.word_embeddings_layernorm.bias'] = ckpt['token_embd_norm.bias']
    state_dict['transformer.ln_f.weight'] = ckpt['output_norm.weight']
    state_dict['transformer.ln_f.bias'] = ckpt['output_norm.bias']
    state_dict['lm_head.weight'] = ckpt['output.weight']
    for i in range(config['bloom.block_count']):
        state_dict[f'transformer.h.{i}.self_attention.query_key_value.weight'] = \
            ckpt[f'blk.{i}.attn_qkv.weight']
        state_dict[f'transformer.h.{i}.self_attention.query_key_value.bias'] = \
            ckpt[f'blk.{i}.attn_qkv.bias']
        state_dict[f'transformer.h.{i}.self_attention.dense.weight'] = \
            ckpt[f'blk.{i}.attn_output.weight']
        state_dict[f'transformer.h.{i}.self_attention.dense.bias'] = \
            ckpt[f'blk.{i}.attn_output.bias']
        state_dict[f'transformer.h.{i}.post_attention_layernorm.weight'] = \
            ckpt[f'blk.{i}.ffn_norm.weight']
        state_dict[f'transformer.h.{i}.post_attention_layernorm.bias'] = \
            ckpt[f'blk.{i}.ffn_norm.bias']
        state_dict[f'transformer.h.{i}.mlp.dense_h_to_4h.weight'] = \
            ckpt[f'blk.{i}.ffn_up.weight']
        state_dict[f'transformer.h.{i}.mlp.dense_h_to_4h.bias'] = \
            ckpt[f'blk.{i}.ffn_up.bias']
        state_dict[f'transformer.h.{i}.mlp.dense_4h_to_h.weight'] = \
            ckpt[f'blk.{i}.ffn_down.weight']
        state_dict[f'transformer.h.{i}.mlp.dense_4h_to_h.bias'] = \
            ckpt[f'blk.{i}.ffn_down.bias']
        state_dict[f'transformer.h.{i}.input_layernorm.weight'] = \
            ckpt[f'blk.{i}.attn_norm.weight']
        state_dict[f'transformer.h.{i}.input_layernorm.bias'] = \
            ckpt[f'blk.{i}.attn_norm.bias']

    with init_empty_weights():
        model = BloomForCausalLM(bloom_config)

    for name, weight in state_dict.items():
        set_module_tensor_to_device(model, name, "cpu", weight, dtype=dtype)
    model = model.cpu()

    pieces, merges = loader.tokenizer_pieces()

    current_directory = os.path.dirname(os.path.abspath(__file__))
    token_file = current_directory + "/model_implement/bloom/tokenizer.json"
    import json
    with open(token_file, 'r') as file:
        data = json.load(file)
    vocab = {}
    # load and replace vocab and merges
    for i in range(len(pieces)):
        token = pieces[i].piece
        score = int(pieces[i].score)
        vocab[token] = score
    data['model']['vocab'] = vocab
    data['model']['merges'] = merges
    with open(token_file, 'w') as file:
        json.dump(data, file)
    tokenizer = BloomTokenizerFast(tokenizer_file=token_file)
    return model, tokenizer


def restore_bloom_weight(ckpt: dict, n_head: int, n_embed: int):
    # see https://github.com/ggerganov/llama.cpp/
    # blob/master/convert-hf-to-gguf.py#L374
    import numpy as np
    for name, weight in ckpt.items():
        if name.endswith("attn_qkv.weight"):
            part1, part2, part3 = np.split(weight.reshape(-1, n_embed), 3, axis=0)
            part1 = part1.reshape((n_head, 1, n_embed // n_head, n_embed))
            part2 = part2.reshape((n_head, 1, n_embed // n_head, n_embed))
            part3 = part3.reshape((n_head, 1, n_embed // n_head, n_embed))
            data = torch.cat([part1, part2, part3], dim=1)
            ckpt[name] = data.reshape(-1, n_embed)
        elif name.endswith("attn_qkv.bias"):
            part1, part2, part3 = np.split(weight, 3, axis=0)
            part1 = part1.reshape((n_head, 1, n_embed // n_head))
            part2 = part2.reshape((n_head, 1, n_embed // n_head))
            part3 = part3.reshape((n_head, 1, n_embed // n_head))
            data = torch.cat([part1, part2, part3], dim=1)
            ckpt[name] = data.reshape(3 * n_embed, )
    return ckpt
