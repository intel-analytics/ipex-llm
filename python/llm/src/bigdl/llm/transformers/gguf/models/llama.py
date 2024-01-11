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
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

from ..gguf import GGUFFileLoader
from bigdl.llm.optimize import optimize_model as optimize_model_fn

import os
import psutil

# 计算占用内存
def show_memory_info(hint):
    pid = os.getpid()
    p = psutil.Process(pid)

    info = p.memory_full_info()
    memory = info.uss / 1024. / 1024
    print('{} memory used: {} MB'.format(hint, memory))

def load_gguf_llama(loader: GGUFFileLoader, dtype: torch.dtype = torch.float, optimize_llm=True,
                    cpu_embedding=False, low_bit='sym_int4'):
    config = loader.config

    llama_config = LlamaConfig(
        vocab_size=len(config['tokenizer.ggml.tokens']),
        hidden_size=config['llama.embedding_length'],
        intermediate_size=config['llama.feed_forward_length'],
        num_hidden_layers=config['llama.block_count'],
        num_attention_heads=config['llama.attention.head_count'],
        num_key_value_heads=config['llama.attention.head_count_kv'],
        hidden_act="silu",
        max_position_embeddings=config['llama.context_length'],
        rms_norm_eps=config['llama.attention.layer_norm_rms_epsilon'],
        use_cache=True,
        pad_token_id=None,
        bos_token_id=config['tokenizer.ggml.bos_token_id'],
        eos_token_id=config['tokenizer.ggml.eos_token_id'],
        pretraining_tp=1,
    )

    n_head = config['llama.attention.head_count']
    n_head_kv = config['llama.attention.head_count_kv']

    with init_empty_weights():
        model = LlamaForCausalLM(llama_config)

    def process_llama(name, tensor):
        nonlocal model
        module_name = get_llama_module_name(name)
        # show_memory_info("3")
        if 'q_proj' in module_name:
            # gguf weight needs to reshape for ffn_gate_inp
            head, hd_size = tensor.shape[0], tensor.shape[1:]
            set_module_tensor_to_device(model, module_name, "cpu", \
                tensor.reshape(n_head, head // n_head // 2, 2, *hd_size)
                                .swapaxes(1, 2)
                                .reshape(tensor.shape), dtype=dtype)
        elif 'k_proj' in module_name:
            head, hd_size = tensor.shape[0], tensor.shape[1:]
            set_module_tensor_to_device(model, module_name, "cpu", \
                tensor.reshape(n_head_kv, head // n_head_kv // 2, 2, *hd_size)
                                .swapaxes(1, 2)
                                .reshape(tensor.shape), dtype=dtype)
        else:
            set_module_tensor_to_device(model, module_name, "cpu", tensor, dtype=dtype)
        # print(model)
        # show_memory_info("4")
        model = optimize_model_fn(model, low_bit=low_bit, optimize_llm=False,
                                    cpu_embedding=cpu_embedding, module_name=module_name, optimize_module=True)
        # show_memory_info("5")

    tensor_loader = loader.tensor_loader
    tensor_loader.load_while_process(process_llama)

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
        tokenizer = LlamaTokenizer(f.name)
        os.remove(f.name)

    return model, tokenizer


def get_llama_module_name(name):
    if name == 'token_embd.weight':
        return 'model.embed_tokens.weight'
    if name == 'output_norm.weight':
        return 'model.norm.weight'
    if name == 'output.weight':
        return 'lm_head.weight'
    layer_id = name.split('.')[1]
    if 'attn_q' in name:
        return  f'model.layers.{layer_id}.self_attn.q_proj.weight'
    if 'attn_k' in name:
        return f'model.layers.{layer_id}.self_attn.k_proj.weight'
    if 'attn_v' in name:
        return f'model.layers.{layer_id}.self_attn.v_proj.weight'
    if 'attn_output' in name:
        return f'model.layers.{layer_id}.self_attn.o_proj.weight'
    if 'ffn_gate' in name:
        return f'model.layers.{layer_id}.mlp.gate_proj.weight'
    if 'ffn_up' in name:
        return f'model.layers.{layer_id}.mlp.up_proj.weight'
    if 'ffn_down' in name:
        return f'model.layers.{layer_id}.mlp.down_proj.weight'
    if 'attn_norm' in name:
        return f'model.layers.{layer_id}.input_layernorm.weight'
    if 'ffn_norm' in name:
        return f'model.layers.{layer_id}.post_attention_layernorm.weight'