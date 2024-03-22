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
from transformers import MptConfig, MptForCausalLM, GPTNeoXTokenizerFast

from ..gguf import GGUFFileLoader


def load_gguf_mpt(loader: GGUFFileLoader, dtype: torch.dtype = torch.float):
    config = loader.config

    mpt_config = MptConfig(
        vocab_size=len(config['tokenizer.ggml.tokens']),
        d_model=config['mpt.embedding_length'],
        n_layers=config['mpt.block_count'],
        n_heads=config['mpt.attention.head_count'],
        max_position_embeddings=config['mpt.context_length'],
        layer_norm_epsilon=config['mpt.attention.layer_norm_epsilon'],
        bos_token_id=config['tokenizer.ggml.bos_token_id'],
        eos_token_id=config['tokenizer.ggml.eos_token_id'],
        unknown_token_id=config['tokenizer.ggml.unknown_token_id'],
    )

    ckpt = loader.tensors(dtype)

    state_dict = {}
    state_dict['transformer.wte.weight'] = ckpt['token_embd.weight']
    state_dict['transformer.norm_f.weight'] = ckpt['output_norm.weight']
    state_dict['lm_head.weight'] = ckpt['output.weight']
    for i in range(config['mpt.block_count']):
        state_dict[f'transformer.blocks.{i}.attn.Wqkv.weight'] = \
            ckpt[f'blk.{i}.attn_qkv.weight']
        state_dict[f'transformer.blocks.{i}.attn.out_proj.weight'] = \
            ckpt[f'blk.{i}.attn_output.weight']
        state_dict[f'transformer.blocks.{i}.norm_2.weight'] = \
            ckpt[f'blk.{i}.ffn_norm.weight']
        state_dict[f'transformer.blocks.{i}.ffn.up_proj.weight'] = \
            ckpt[f'blk.{i}.ffn_up.weight']
        state_dict[f'transformer.blocks.{i}.ffn.down_proj.weight'] = \
            ckpt[f'blk.{i}.ffn_down.weight']
        state_dict[f'transformer.blocks.{i}.norm_1.weight'] = \
            ckpt[f'blk.{i}.attn_norm.weight']

    with init_empty_weights():
        model = MptForCausalLM(mpt_config)

    for name, weight in state_dict.items():
        set_module_tensor_to_device(model, name, "cpu", weight, dtype=dtype)

    model = model.cpu()

    pieces, merges = loader.tokenizer_pieces()

    current_directory = os.path.dirname(os.path.abspath(__file__))
    token_file = current_directory + "/model_implement/mpt/tokenizer.json"
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
    tokenizer = GPTNeoXTokenizerFast(tokenizer_file=token_file)
    return model, tokenizer
