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
from transformers import LlamaTokenizer

from ..gguf import GGUFFileLoader
from .model_implement.yuan2.yuan_hf_model import YuanForCausalLM
from .model_implement.yuan2.configuration_yuan import YuanConfig


def load_gguf_yuan(loader: GGUFFileLoader, dtype: torch.dtype = torch.float,
                   low_bit='sym_int4'):
    config = loader.config
    yuan_config = YuanConfig(
        vocab_size=len(config['tokenizer.ggml.tokens']),
        hidden_size=config['llama.embedding_length'],
        intermediate_size=config['llama.feed_forward_length'],
        num_hidden_layers=config['llama.block_count'],
        num_attention_heads=config['llama.attention.head_count'],
        hidden_act="silu",
        model_max_length=config['llama.context_length'],
        rms_norm_eps=config['llama.attention.layer_norm_rms_epsilon'],
        use_cache=True,
        initializer_range=0.02,
        pad_token_id=config['tokenizer.ggml.padding_token_id'],
        bos_token_id=config['tokenizer.ggml.bos_token_id'],
        eos_token_id=config['tokenizer.ggml.eos_token_id'],
        eod_token=config['tokenizer.ggml.eos_token_id'],
        eod_token_id=config['tokenizer.ggml.eos_token_id'],
        sep_token=config['tokenizer.ggml.seperator_token_id'],
        sep_token_id=config['tokenizer.ggml.seperator_token_id'],
        mask_token_id=config['tokenizer.ggml.seperator_token_id'],
        use_loss_mask=False,
        dropout=0.1,
        reset_attention_mask=True,
        reset_position_ids=True,
        max_position_embeddings=config['llama.context_length'],
        causal_mask=True,
        use_flash_attention=False,
        pretraining_tp=1,
    )

    ckpt = loader.tensors(dtype)
    n_head = config['llama.attention.head_count']
    n_head_kv = config['llama.attention.head_count_kv']

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
        state_dict[f'model.layers.{i}.self_attn.lf_gate.output_layernorm.weight'] = \
            ckpt[f'blk.{i}.lf_output_norm.weight']
        state_dict[f'model.layers.{i}.self_attn.lf_gate.conv1.weight'] = \
            ckpt[f'blk.{i}.conv1.weight']
        state_dict[f'model.layers.{i}.self_attn.lf_gate.conv2.weight'] = \
            ckpt[f'blk.{i}.conv2.weight']
        state_dict[f'model.layers.{i}.self_attn.lf_gate.conv1.bias'] = \
            ckpt[f'blk.{i}.conv1.bias']
        state_dict[f'model.layers.{i}.self_attn.lf_gate.conv2.bias'] = \
            ckpt[f'blk.{i}.conv2.bias']

    with init_empty_weights():
        model = YuanForCausalLM(yuan_config).eval()

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
        tokenizer = LlamaTokenizer(f.name)
        os.remove(f.name)

    tokenizer.add_eos_token = False
    tokenizer.add_bos_token = False
    tokenizer.eos_token = '<eod>'

    tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>', '<commit_before>',  # noqa
                         '<commit_msg>', '<commit_after>', '<jupyter_start>', '<jupyter_text>', '<jupyter_code>', '<jupyter_output>', '<empty_output>'], special_tokens=True)  # noqa

    return model, tokenizer
