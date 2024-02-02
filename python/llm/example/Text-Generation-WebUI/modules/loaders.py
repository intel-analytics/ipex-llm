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

# This file is adapted from
# https://github.com/oobabooga/text-generation-webui/blob/main/modules/loaders.py


import functools
from collections import OrderedDict

import gradio as gr

from modules import shared

loaders_and_params = OrderedDict({
    'Transformers': [
        'cpu_memory',
        'gpu_memory',
        'load_in_8bit',
        'bf16',
        'cpu',
        'disk',
        'auto_devices',
        'load_in_4bit',
        'use_double_quant',
        'quant_type',
        'compute_dtype',
        'trust_remote_code',
        'no_use_fast',
        'use_flash_attention_2',
        'alpha_value',
        'rope_freq_base',
        'compress_pos_emb',
        'disable_exllama',
        'disable_exllamav2',
        'transformers_info'
    ],
    'llama.cpp': [
        'n_ctx',
        'n_gpu_layers',
        'tensor_split',
        'n_batch',
        'threads',
        'threads_batch',
        'no_mmap',
        'mlock',
        'no_mul_mat_q',
        'alpha_value',
        'rope_freq_base',
        'compress_pos_emb',
        'cpu',
        'numa',
        'no_offload_kqv',
        'tensorcores',
    ],
    'llamacpp_HF': [
        'n_ctx',
        'n_gpu_layers',
        'tensor_split',
        'n_batch',
        'threads',
        'threads_batch',
        'no_mmap',
        'mlock',
        'no_mul_mat_q',
        'alpha_value',
        'rope_freq_base',
        'compress_pos_emb',
        'cpu',
        'numa',
        'cfg_cache',
        'trust_remote_code',
        'no_use_fast',
        'logits_all',
        'no_offload_kqv',
        'tensorcores',
        'llamacpp_HF_info',
    ],
    'ExLlamav2_HF': [
        'gpu_split',
        'max_seq_len',
        'cfg_cache',
        'no_flash_attn',
        'num_experts_per_token',
        'cache_8bit',
        'alpha_value',
        'compress_pos_emb',
        'trust_remote_code',
        'no_use_fast',
    ],
    'ExLlamav2': [
        'gpu_split',
        'max_seq_len',
        'no_flash_attn',
        'num_experts_per_token',
        'cache_8bit',
        'alpha_value',
        'compress_pos_emb',
        'exllamav2_info',
    ],
    'AutoGPTQ': [
        'triton',
        'no_inject_fused_attention',
        'no_inject_fused_mlp',
        'no_use_cuda_fp16',
        'wbits',
        'groupsize',
        'desc_act',
        'disable_exllama',
        'disable_exllamav2',
        'gpu_memory',
        'cpu_memory',
        'cpu',
        'disk',
        'auto_devices',
        'trust_remote_code',
        'no_use_fast',
        'autogptq_info',
    ],
    'BigDL-LLM': [
        'load_in_4bit',
        'load_in_low_bit',
        'optimize_model',
        #'modules_to_not_convert',
        'cpu_embedding',
        #'lightweight_bmm',
        'trust_remote_code',
        'use_cache',
    ],
    'AutoAWQ': [
        'cpu_memory',
        'gpu_memory',
        'auto_devices',
        'max_seq_len',
        'no_inject_fused_attention',
        'trust_remote_code',
        'no_use_fast',
    ],
    'GPTQ-for-LLaMa': [
        'wbits',
        'groupsize',
        'model_type',
        'pre_layer',
        'trust_remote_code',
        'no_use_fast',
        'gptq_for_llama_info',
    ],
    'ctransformers': [
        'n_ctx',
        'n_gpu_layers',
        'n_batch',
        'threads',
        'model_type',
        'no_mmap',
        'mlock'
    ],
    'QuIP#': [
        'trust_remote_code',
        'no_use_fast',
        'no_flash_attn',
        'quipsharp_info',
    ],
    'HQQ': [
        'hqq_backend',
        'trust_remote_code',
        'no_use_fast',
    ]
})


def transformers_samplers():
    return {
        'temperature',
        'temperature_last',
        'dynamic_temperature',
        'dynamic_temperature_low',
        'top_p',
        'min_p',
        'top_k',
        'typical_p',
        'epsilon_cutoff',
        'eta_cutoff',
        'tfs',
        'top_a',
        'repetition_penalty',
        'presence_penalty',
        'frequency_penalty',
        'repetition_penalty_range',
        'encoder_repetition_penalty',
        'no_repeat_ngram_size',
        'min_length',
        'seed',
        'do_sample',
        'penalty_alpha',
        'num_beams',
        'length_penalty',
        'early_stopping',
        'mirostat_mode',
        'mirostat_tau',
        'mirostat_eta',
        'grammar_file_row',
        'grammar_string',
        'guidance_scale',
        'negative_prompt',
        'ban_eos_token',
        'custom_token_bans',
        'add_bos_token',
        'skip_special_tokens',
        'auto_max_new_tokens',
    }


loaders_samplers = {
    'Transformers': transformers_samplers(),
    'AutoGPTQ': transformers_samplers(),
    'GPTQ-for-LLaMa': transformers_samplers(),
    'AutoAWQ': transformers_samplers(),
    'QuIP#': transformers_samplers(),
    'HQQ': transformers_samplers(),
    'BigDL-LLM': transformers_samplers(),
    'ExLlamav2': {
        'temperature',
        'top_p',
        'min_p',
        'top_k',
        'typical_p',
        'tfs',
        'repetition_penalty',
        'repetition_penalty_range',
        'seed',
        'mirostat_mode',
        'mirostat_tau',
        'mirostat_eta',
        'ban_eos_token',
        'add_bos_token',
        'custom_token_bans',
        'skip_special_tokens',
        'auto_max_new_tokens',
    },
    'ExLlamav2_HF': {
        'temperature',
        'temperature_last',
        'dynamic_temperature',
        'dynamic_temperature_low',
        'top_p',
        'min_p',
        'top_k',
        'typical_p',
        'epsilon_cutoff',
        'eta_cutoff',
        'tfs',
        'top_a',
        'repetition_penalty',
        'presence_penalty',
        'frequency_penalty',
        'repetition_penalty_range',
        'encoder_repetition_penalty',
        'no_repeat_ngram_size',
        'min_length',
        'seed',
        'do_sample',
        'mirostat_mode',
        'mirostat_tau',
        'mirostat_eta',
        'grammar_file_row',
        'grammar_string',
        'guidance_scale',
        'negative_prompt',
        'ban_eos_token',
        'custom_token_bans',
        'add_bos_token',
        'skip_special_tokens',
        'auto_max_new_tokens',
    },
    'llama.cpp': {
        'temperature',
        'top_p',
        'min_p',
        'top_k',
        'typical_p',
        'tfs',
        'repetition_penalty',
        'presence_penalty',
        'frequency_penalty',
        'seed',
        'mirostat_mode',
        'mirostat_tau',
        'mirostat_eta',
        'grammar_file_row',
        'grammar_string',
        'ban_eos_token',
        'custom_token_bans',
    },
    'llamacpp_HF': {
        'temperature',
        'temperature_last',
        'dynamic_temperature',
        'dynamic_temperature_low',
        'top_p',
        'min_p',
        'top_k',
        'typical_p',
        'epsilon_cutoff',
        'eta_cutoff',
        'tfs',
        'top_a',
        'repetition_penalty',
        'presence_penalty',
        'frequency_penalty',
        'repetition_penalty_range',
        'encoder_repetition_penalty',
        'no_repeat_ngram_size',
        'min_length',
        'seed',
        'do_sample',
        'mirostat_mode',
        'mirostat_tau',
        'mirostat_eta',
        'grammar_file_row',
        'grammar_string',
        'guidance_scale',
        'negative_prompt',
        'ban_eos_token',
        'custom_token_bans',
        'add_bos_token',
        'skip_special_tokens',
        'auto_max_new_tokens',
    },
    'ctransformers': {
        'temperature',
        'top_p',
        'top_k',
        'repetition_penalty',
        'repetition_penalty_range',
    },
}

loaders_model_types = {
    'GPTQ-for-LLaMa': [
        "None",
        "llama",
        "opt",
        "gptj"
    ],
    'ctransformers': [
        "None",
        "gpt2",
        "gptj",
        "gptneox",
        "llama",
        "mpt",
        "dollyv2",
        "replit",
        "starcoder",
        "gptbigcode",
        "falcon"
    ],
}


@functools.cache
def list_all_samplers():
    all_samplers = set()
    for k in loaders_samplers:
        for sampler in loaders_samplers[k]:
            all_samplers.add(sampler)

    return sorted(all_samplers)


def blacklist_samplers(loader):
    all_samplers = list_all_samplers()
    if loader == 'All':
        return [gr.update(visible=True) for sampler in all_samplers]
    else:
        return [gr.update(visible=True) if sampler in loaders_samplers[loader] else gr.update(visible=False) for sampler in all_samplers]


def get_model_types(loader):
    if loader in loaders_model_types:
        return loaders_model_types[loader]

    return ["None"]


def get_gpu_memory_keys():
    return [k for k in shared.gradio if k.startswith('gpu_memory')]


@functools.cache
def get_all_params():
    all_params = set()
    for k in loaders_and_params:
        for el in loaders_and_params[k]:
            all_params.add(el)

    if 'gpu_memory' in all_params:
        all_params.remove('gpu_memory')
        for k in get_gpu_memory_keys():
            all_params.add(k)

    return sorted(all_params)


def make_loader_params_visible(loader):
    params = []
    all_params = get_all_params()
    if loader in loaders_and_params:
        params = loaders_and_params[loader]

        if 'gpu_memory' in params:
            params.remove('gpu_memory')
            params += get_gpu_memory_keys()

    return [gr.update(visible=True) if k in params else gr.update(visible=False) for k in all_params]
