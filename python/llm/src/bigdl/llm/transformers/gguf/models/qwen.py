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

from .model_implement.qwen.configuration_qwen import QWenConfig
from .model_implement.qwen.modeling_qwen import QWenLMHeadModel
from .model_implement.qwen.tokenization_qwen import QWenTokenizer
from ..gguf import GGUFFileLoader


def load_gguf_qwen(loader: GGUFFileLoader, dtype: torch.dtype = torch.float):
    config = loader.config

    qwen_config = QWenConfig(
        vocab_size=len(config['tokenizer.ggml.tokens']),
        hidden_size=config['qwen.embedding_length'],
        intermediate_size=config['qwen.feed_forward_length'],
        num_hidden_layers=config['qwen.block_count'],
        num_attention_heads=config['qwen.attention.head_count'],
        rotary_emb_base=config['qwen.rope.freq_base'],
        max_position_embeddings=config['qwen.context_length'],
        layer_norm_epsilon=config['qwen.attention.layer_norm_rms_epsilon'],
        pad_token_id=None,
        bos_token_id=config['tokenizer.ggml.bos_token_id'],
        eos_token_id=config['tokenizer.ggml.eos_token_id'],
        seq_length=8192,
    )

    ckpt = loader.tensors(dtype)

    state_dict = {}
    state_dict['transformer.wte.weight'] = ckpt['token_embd.weight']
    state_dict['transformer.ln_f.weight'] = ckpt['output_norm.weight']
    state_dict['lm_head.weight'] = ckpt['output.weight']
    for i in range(config['qwen.block_count']):
        state_dict[f'transformer.h.{i}.attn.c_attn.weight'] = \
            ckpt[f'blk.{i}.attn_qkv.weight']
        state_dict[f'transformer.h.{i}.attn.c_attn.bias'] = \
            ckpt[f'blk.{i}.attn_qkv.bias']
        state_dict[f'transformer.h.{i}.attn.c_proj.weight'] = \
            ckpt[f'blk.{i}.attn_output.weight']
        state_dict[f'transformer.h.{i}.mlp.w2.weight'] = \
            ckpt[f'blk.{i}.ffn_gate.weight']
        state_dict[f'transformer.h.{i}.mlp.w1.weight'] = \
            ckpt[f'blk.{i}.ffn_up.weight']
        state_dict[f'transformer.h.{i}.mlp.c_proj.weight'] = \
            ckpt[f'blk.{i}.ffn_down.weight']
        state_dict[f'transformer.h.{i}.ln_1.weight'] = \
            ckpt[f'blk.{i}.attn_norm.weight']
        state_dict[f'transformer.h.{i}.ln_2.weight'] = \
            ckpt[f'blk.{i}.ffn_norm.weight']

    with init_empty_weights():
        model = QWenLMHeadModel(qwen_config)

    for name, weight in state_dict.items():
        set_module_tensor_to_device(model, name, "cpu", weight)

    model = model.cpu()


    pieces = loader.tokenizer_pieces()

    output_temp_file = "./temp_qwen_tokenizer.tmp"
    with open(output_temp_file, 'w', encoding='utf-8') as f:
        import base64
        # see https://github.com/simonJJJ/llama.cpp/
        # blob/ad04d174f623711b15bdddd16050b7af9db9ef0b/convert-hf-to-gguf.py#L889
        for i in range(len(pieces)):
            token = pieces[i].piece
            score = int(pieces[i].score)
            byte = string_to_token_bytes(token)
            line = base64.b64encode(byte).decode("utf8") + " " + str(score) + "\n"
            f.write(line)
        f.close()
        tokenizer = QWenTokenizer(f.name)
        os.remove(f.name)

    return model, tokenizer


def string_to_token_bytes(s):
    # see https://github.com/simonJJJ/llama.cpp/blob/
    # ad04d174f623711b15bdddd16050b7af9db9ef0b/convert-hf-to-gguf.py#L842
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
    byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
    return bytes([byte_decoder[char] for char in s])
