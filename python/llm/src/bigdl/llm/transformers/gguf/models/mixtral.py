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
from transformers import MixtralConfig, MixtralForCausalLM, LlamaTokenizer

from ..gguf import GGUFFileLoader


def load_gguf_mixtral(loader: GGUFFileLoader, dtype: torch.dtype = torch.float):
    # mixtral enjoys a general architecture of llma
    # e.g. it applies llama tokenizer
    config = loader.config
    num_local_experts = config['llama.expert_count']
    num_experts_per_tok = config['llama.expert_used_count']
    n_head = config['llama.attention.head_count']
    n_head_kv = config['llama.attention.head_count_kv']
    hidden_size = config['llama.embedding_length']

    mixtral_config = MixtralConfig(
        vocab_size=len(config['tokenizer.ggml.tokens']),
        hidden_size=hidden_size,
        intermediate_size=config['llama.feed_forward_length'],
        num_hidden_layers=config['llama.block_count'],
        num_attention_heads=config['llama.attention.head_count'],
        num_key_value_heads=config['llama.attention.head_count_kv'],
        max_position_embeddings=config['llama.context_length'],
        rms_norm_eps=config['llama.attention.layer_norm_rms_epsilon'],
        pad_token_id=config['tokenizer.ggml.padding_token_id'],
        bos_token_id=config['tokenizer.ggml.bos_token_id'],
        eos_token_id=config['tokenizer.ggml.eos_token_id'],
        num_local_experts=num_local_experts,
        num_experts_per_tok=num_experts_per_tok,
    )

    ckpt = loader.tensors(dtype)
    from .llama import restore_llama_weight
    ckpt = restore_llama_weight(ckpt, n_head, n_head_kv)

    state_dict = {}
    state_dict['model.embed_tokens.weight'] = ckpt['token_embd.weight']
    state_dict['model.norm.weight'] = ckpt['output_norm.weight']
    state_dict['lm_head.weight'] = ckpt['output.weight']
    for i in range(config['llama.block_count']):
        state_dict[f'model.layers.{i}.self_attn.q_proj.weight'] = \
            ckpt[f'blk.{i}.attn_q.weight']
        state_dict[f'model.layers.{i}.self_attn.k_proj.weight'] = \
            ckpt[f'blk.{i}.attn_k.weight']
        state_dict[f'model.layers.{i}.self_attn.v_proj.weight'] = \
            ckpt[f'blk.{i}.attn_v.weight']
        state_dict[f'model.layers.{i}.self_attn.o_proj.weight'] = \
            ckpt[f'blk.{i}.attn_output.weight']
        state_dict[f'model.layers.{i}.input_layernorm.weight'] = \
            ckpt[f'blk.{i}.attn_norm.weight']
        state_dict[f'model.layers.{i}.post_attention_layernorm.weight'] = \
            ckpt[f'blk.{i}.ffn_norm.weight']
        state_dict[f'model.layers.{i}.block_sparse_moe.gate.weight'] = \
            ckpt[f'blk.{i}.ffn_gate_inp.weight'].reshape(num_local_experts, hidden_size)
        for j in range(num_local_experts):
            state_dict[f'model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight'] = \
                (ckpt[f'blk.{i}.ffn_gate.{j}.weight'])
            state_dict[f'model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight'] = \
                ckpt[f'blk.{i}.ffn_down.{j}.weight']
            state_dict[f'model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight'] = \
                ckpt[f'blk.{i}.ffn_up.{j}.weight']

    with init_empty_weights():
        model = MixtralForCausalLM(mixtral_config)

    for name, weight in state_dict.items():
        set_module_tensor_to_device(model, name, "cpu", weight, dtype=dtype)

    model = model.cpu()

    from transformers.convert_slow_tokenizer import import_protobuf
    spm_pb2 = import_protobuf("Failed to import protobuf")

    tokenizer_pieces = loader.tokenizer_pieces()
    trainer_spec = spm_pb2.TrainerSpec(byte_fallback=True,
                                       model_type=spm_pb2.TrainerSpec.ModelType.BPE)
    proto = spm_pb2.ModelProto(pieces=tokenizer_pieces, trainer_spec=trainer_spec)
    proto = proto.SerializeToString()

    with NamedTemporaryFile(delete=False) as f:
        f.write(proto)
        f.close()
        tokenizer = LlamaTokenizer(f.name)
        os.remove(f.name)

    return model, tokenizer
