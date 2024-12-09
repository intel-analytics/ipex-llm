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
from safetensors.torch import load_file
from tokenizers import processors

from transformers import LlamaConfig, PreTrainedTokenizerFast
from ipex_llm.transformers.utils import invalidInputError


VIT_KEY = "vit_path"  # FIXME: just made at random
VIT_FILE = "vit_adapter.pt"


def load_weights(input_dir: str):
    safetensor_files = [os.path.join(input_dir, x) for x in os.listdir(input_dir)
                        if x.endswith(".safetensors")]
    bin_files = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if x.endswith(".bin")]

    all_weights = {}

    if safetensor_files:
        if len(safetensor_files) > 1:
            safetensor_files = sorted(safetensor_files, key=lambda x: int(x.rsplit("-", 3)[1]))
        for file in safetensor_files:
            tensors = load_file(file)
            all_weights.update(tensors)
        return all_weights

    elif bin_files:
        if len(bin_files) > 1:
            bin_files = sorted(bin_files, key=lambda x: int(x.rsplit("-", 3)[1]))
        for file in bin_files:
            tensors = torch.load(file, map_location="cpu")
            all_weights.update(tensors)
        return all_weights

    else:
        invalidInputError(False, "No .safetensors or .bin files found in the specified directory.")


def convert_state_dict(original_state_dict: dict, config: LlamaConfig,
                       partial_rotary_factor: float, decouple_tied_embeddings=False):
    hidden_size, num_heads = config.hidden_size, config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = hidden_size // num_heads
    rotary_dim = int(partial_rotary_factor * head_dim)
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))

    # permute for sliced rotary
    def permute_weight(w, num_heads, rotary_dim):
        w = w.view(num_heads, head_dim, hidden_size)
        w, w_pass = w[:, :rotary_dim, :], w[:, rotary_dim:, :]
        w = w.view(num_heads, rotary_dim // 2, 2, hidden_size).transpose(1, 2)\
            .reshape(num_heads, rotary_dim, hidden_size)
        return torch.cat([w, w_pass], dim=1).view(num_heads * head_dim, hidden_size)

    def permute_bias(b, num_heads, rotary_dim):
        b = b.view(num_heads, head_dim)
        b, b_pass = b[:, :rotary_dim], b[:, rotary_dim:]
        b = b.view(num_heads, rotary_dim // 2, 2).transpose(1, 2).reshape(num_heads, rotary_dim)
        return torch.cat([b, b_pass], dim=1).view(num_heads * head_dim)

    new_dict, vit_dict = {}, {}
    param_count = 0
    index_dict = {"weight_map": {}}
    for key, value in original_state_dict.items():
        if "model.vision" in key:  # vit
            vit_dict[key.replace("model.vision.", "")] = value.detach().clone()
        elif "q_proj." in key:
            if "weight" in key:
                new_dict[key] = permute_weight(value, num_heads, rotary_dim)
            elif config.attention_bias:  # bias
                new_dict[key] = permute_bias(value, num_heads, rotary_dim)
        elif "k_proj." in key:
            if "weight" in key:
                new_dict[key] = permute_weight(value, num_key_value_heads, rotary_dim)
            elif config.attention_bias:  # bias
                new_dict[key] = permute_bias(value, num_key_value_heads, rotary_dim)
        elif "v_proj." in key:
            if "bias" in key and not config.attention_bias:
                continue
            new_dict[key] = value
        elif "o_proj." in key:
            new_dict[key] = value
            if config.attention_bias:  # bias
                new_dict[key.replace("weight", "bias")] = torch.zeros(hidden_size,
                                                                      dtype=value.dtype)
        elif "gate_up_proj." in key:
            gate_proj, up_proj = value.chunk(2, dim=0)
            new_dict[key.replace("gate_up_proj.", "gate_proj.")] = gate_proj
            new_dict[key.replace("gate_up_proj.", "up_proj.")] = up_proj
        else:
            new_dict[key] = value

    for layer_i in range(config.num_hidden_layers):
        new_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq.clone()

    if decouple_tied_embeddings:
        new_dict["transformer.output_layer.weight"] = \
            original_state_dict["model.embed_tokens.weight"].clone()

    return new_dict, vit_dict


def convert_config(original_config: dict, decouple_tied_embeddings=False):
    similar_keys_to_keep = [
        "num_attention_heads",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "rms_norm_eps",
        "num_key_value_heads",
        "vocab_size",
        "partial_rotary_factor",
        "rope_theta",
        "max_position_embeddings",
        "attention_bias",
        "torch_dtype",
        "tie_word_embeddings",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "boi_token_id",
        "eoi_token_id",
        "vision_config",
    ]
    new_config_kwargs = {k: v for k, v in original_config.items() if k in similar_keys_to_keep}
    if getattr(original_config, "partial_rotary_factor", 1) < 1:
        new_config_kwargs["rope_dim"] = original_config["head_dim"] * \
            original_config["partial_rotary_factor"]
    if decouple_tied_embeddings:
        new_config_kwargs["tie_word_embeddings"] = False
    if "vision_config" in original_config:
        new_config_kwargs["vision_config"] = original_config["vision_config"]
        new_config_kwargs[VIT_KEY] = VIT_FILE
    if "bos_token_id" not in new_config_kwargs:
        new_config_kwargs["bos_token_id"] = None

    new_config = LlamaConfig(**new_config_kwargs)
    return new_config


def convert_glm_tokenizer(input_dir):
    fast_tok = PreTrainedTokenizerFast.from_pretrained(input_dir,
                                                       model_input_names=["input_ids",
                                                                          "attention_mask"])
    fast_tok._tokenizer.post_processor = processors.Sequence(
        [processors.ByteLevel(trim_offsets=False)],
    )
    return fast_tok
