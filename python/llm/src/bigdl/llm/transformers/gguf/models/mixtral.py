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
from bigdl.llm.ggml.quantize import ggml_tensor_qtype
from bigdl.llm.transformers.convert import replace_with_low_bit_linear_for_module


def load_gguf_mixtral(loader: GGUFFileLoader, dtype: torch.dtype = torch.float,
                      low_bit='sym_int4'):
    # mixtral enjoys a general architecture of llma
    # e.g. it applies llama tokenizer
    config = loader.config
    num_local_experts = config['llama.expert_count']
    num_experts_per_tok = config['llama.expert_used_count']
    n_head = config['llama.attention.head_count']
    n_head_kv = config['llama.attention.head_count_kv']
    hidden_size = config['llama.embedding_length']
    qtype = ggml_tensor_qtype[low_bit]

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

    with init_empty_weights():
        model = MixtralForCausalLM(mixtral_config)

    # define an operator function that passed to low-level gguf API
    def process_mixtral(name, tensor):
        nonlocal model
        # prepare module's name in transformers
        module_name = get_mixtral_module_name(name)
        # prepare module's weight in transformers
        if 'ffn_gate_inp' in name:
            tensor = tensor.reshape(num_local_experts, hidden_size)
        elif name.endswith("attn_q.weight"):
            head, hd_size = tensor.shape[0], tensor.shape[1:]
            tensor = (tensor.reshape(n_head,
                                     head // n_head // 2,
                                     2,
                                     *hd_size)
                            .swapaxes(1, 2)
                            .reshape(tensor.shape))
        elif name.endswith("attn_k.weight"):
            head, hd_size = tensor.shape[0], tensor.shape[1:]
            tensor = (tensor.reshape(n_head_kv,
                                     head // n_head_kv // 2,
                                     2,
                                     *hd_size)
                            .swapaxes(1, 2)
                            .reshape(tensor.shape))
        set_module_tensor_to_device(model,
                                    module_name,
                                    "cpu",
                                    tensor,
                                    dtype=dtype)
        model = replace_with_low_bit_linear_for_module(model, qtype=qtype, module_name=module_name)

    tensor_loader = loader.tensor_loader
    tensor_loader.load_while_process(process_mixtral)

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


def get_mixtral_module_name(name):
        if name == 'token_embd.weight':
            return 'model.embed_tokens.weight'
        if name == 'output_norm.weight':
            return 'model.norm.weight'
        if name == 'output.weight':
            return 'lm_head.weight'
        layer_id = name.split('.')[1]
        if 'attn_q' in name:
            return f'model.layers.{layer_id}.self_attn.q_proj.weight'
        if 'attn_k' in name:
            return f'model.layers.{layer_id}.self_attn.k_proj.weight'
        if 'attn_v' in name:
            return f'model.layers.{layer_id}.self_attn.v_proj.weight'
        if 'attn_output' in name:
            return f'model.layers.{layer_id}.self_attn.o_proj.weight'
        if 'attn_norm' in name:
            return f'model.layers.{layer_id}.input_layernorm.weight'
        if 'ffn_norm' in name:
            return f'model.layers.{layer_id}.post_attention_layernorm.weight'
        if 'ffn_gate_inp' in name:
            return f'model.layers.{layer_id}.block_sparse_moe.gate.weight'
        local_expert_id = name.split('.')[3]
        if 'ffn_gate' in name:
            return f'model.layers.{layer_id}.' + \
                   f'block_sparse_moe.experts.{local_expert_id}.w1.weight'
        if 'ffn_down' in name:
            return f'model.layers.{layer_id}.' + \
                   f'block_sparse_moe.experts.{local_expert_id}.w2.weight'
        if 'ffn_up' in name:
            return f'model.layers.{layer_id}.' + \
                   f'block_sparse_moe.experts.{local_expert_id}.w3.weight'
