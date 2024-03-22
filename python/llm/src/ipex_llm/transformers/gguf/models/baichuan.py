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
from ipex_llm.ggml.quantize import ggml_tensor_qtype
from ipex_llm.transformers.convert import replace_with_low_bit_linear_for_module


def load_gguf_baichuan(loader: GGUFFileLoader, dtype: torch.dtype = torch.float,
                       low_bit='sym_int4'):
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

    qtype = ggml_tensor_qtype[low_bit]
    n_head = config['baichuan.attention.head_count']
    n_head_kv = config['baichuan.attention.head_count_kv']

    with init_empty_weights():
        model = BaiChuanForCausalLM(baichuan_config)

    attn_q_tensor, attn_k_tensor, attn_v_tensor = [torch.tensor([]) for _ in range(3)]

    def process_baichuan(name, tensor):
        nonlocal model, attn_q_tensor, attn_k_tensor, attn_v_tensor
        module_name = get_baichuan_module_name(name)
        tensor = restore_baichuan_weight(name, tensor, n_head, n_head_kv)

        if 'attn_q' in name:
            attn_q_tensor = tensor
            return
        if 'attn_k' in name:
            attn_k_tensor = tensor
            return
        if 'attn_v' in name:
            attn_v_tensor = tensor
            tensor = torch.cat([attn_q_tensor, attn_k_tensor, attn_v_tensor], dim=0)
        set_module_tensor_to_device(model, module_name, "cpu", tensor, dtype=dtype)
        if 'lm_head' in module_name:
            return
        model = replace_with_low_bit_linear_for_module(model, qtype=qtype, module_name=module_name)

    tensor_loader = loader.tensor_loader
    tensor_loader.load_while_process(process_baichuan)

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


def restore_baichuan_weight(name, weight, n_head: int, n_head_kv: int):
    # see https://github.com/ggerganov/llama.cpp/blob/master/convert-hf-to-gguf.py#L535

    head, hd_size = weight.shape[0], weight.shape[1:]
    if n_head != n_head_kv:
        new_n_head = n_head // n_head_kv
    else:
        new_n_head = n_head
    if name.endswith("attn_q.weight"):
        weight = (weight.reshape(new_n_head, head // new_n_head // 2, 2, *hd_size)
                        .swapaxes(1, 2)
                        .reshape(weight.shape))
    elif name.endswith("attn_k.weight"):
        weight = (weight.reshape(new_n_head, head // new_n_head // 2, 2, *hd_size)
                        .swapaxes(1, 2)
                        .reshape(weight.shape))
    return weight


def get_baichuan_module_name(name):
    if name == 'token_embd.weight':
        return 'model.embed_tokens.weight'
    if name == 'output_norm.weight':
        return 'model.norm.weight'
    if name == 'output.weight':
        return 'lm_head.weight'
    layer_id = name.split('.')[1]
    if 'attn_q' in name or 'attn_k' in name or 'attn_v' in name:
        return f'model.layers.{layer_id}.self_attn.W_pack.weight'
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
