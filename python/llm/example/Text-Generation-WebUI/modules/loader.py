import functools
from collections import OrderedDict

import gradio as gr

from modules import shared

loaders_and_params = OrderedDict({
    'BigDL-LLM': [
        'load_in_4bit',
        'load_in_low_bit',
        'optimize_model',
        'modules_to_not_convert',
        'cpu_embedding',
        'lightweight_bmm',
        'trust_remote_code',
        'use_cache',
    ],
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
    'BigDL-LLM': transformers_samplers(),
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
