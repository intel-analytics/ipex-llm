import gc
import logging
import os
import re
import time
import traceback
from pathlib import Path

import torch
import transformers
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate.utils import is_ccl_available, is_xpu_available
from transformers import (
    AutoConfig,
    AutoTokenizer,
)

import modules.shared as shared
from modules.logging_colors import logger
from modules.models_settings import get_model_metadata

transformers.logging.set_verbosity_error()

local_rank = None

def load_model(model_name, loader=None):
    logger.info(f"Loading {model_name}")
    t0 = time.time()

    shared.is_seq2seq = False
    shared.model_name = model_name
    load_func_map = {
        'BigDL-LLM': bigdl_llm_loader,
    }

    metadata = get_model_metadata(model_name)
    if loader is None:
        if shared.args.loader is not None:
            loader = shared.args.loader
        else:
            loader = metadata['loader']
            if loader is None:
                logger.error('The path to the model does not exist. Exiting.')
                raise ValueError

    shared.args.loader = loader
    output = load_func_map[loader](model_name)
    if type(output) is tuple:
        model, tokenizer = output
    else:
        model = output
        if model is None:
            return None, None
        else:
            tokenizer = load_tokenizer(model_name, model)

    shared.settings.update({k: v for k, v in metadata.items() if k in shared.settings})

    logger.info(f"LOADER: {loader}")
    logger.info(f"TRUNCATION LENGTH: {shared.settings['truncation_length']}")
    logger.info(f"INSTRUCTION TEMPLATE: {metadata['instruction_template']}")
    logger.info(f"Loaded the model in {(time.time()-t0):.2f} seconds.")
    return model, tokenizer


def load_tokenizer(model_name, model):
    tokenizer = None
    path_to_model = Path(f"{shared.args.model_dir}/{model_name}/")
    if any(s in model_name.lower() for s in ['gpt-4chan', 'gpt4chan']) and Path(f"{shared.args.model_dir}/gpt-j-6B/").exists():
        tokenizer = AutoTokenizer.from_pretrained(Path(f"{shared.args.model_dir}/gpt-j-6B/"))
    elif path_to_model.exists():
        if shared.args.no_use_fast:
            logger.info('Loading the tokenizer with use_fast=False.')

        tokenizer = AutoTokenizer.from_pretrained(
            path_to_model,
            trust_remote_code=shared.args.trust_remote_code,
            use_fast=not shared.args.no_use_fast
        )

    return tokenizer


def bigdl_llm_loader(model_name):

    from bigdl.llm.transformers import AutoModelForCausalLM, AutoModel, AutoModelForSeq2SeqLM

    path_to_model = Path(f'{shared.args.model_dir}/{model_name}')

    config = AutoConfig.from_pretrained(path_to_model, trust_remote_code=shared.args.trust_remote_code)

    if 'chatglm' in model_name.lower():
        LoaderClass = AutoModel
    else:
        if config.to_dict().get('is_encoder_decoder', False):
            LoaderClass = AutoModelForSeq2SeqLM
            shared.is_seq2seq = True
        else:
            LoaderClass = AutoModelForCausalLM

    model = LoaderClass.from_pretrained(
                path_to_model,
                load_in_4bit=shared.args.load_in_4bit,
                load_in_low_bit=shared.args.load_in_low_bit,
                optimize_model=shared.args.optimize_model,
                modules_to_not_convert=shared.args.modules_to_not_convert,
                cpu_embedding=shared.args.cpu_embedding,
                lightweight_bmm=shared.args.lightweight_bmm,
                trust_remote_code=shared.args.trust_remote_code,
                use_cache=shared.args.use_cache,
                )

    if shared.args.device == "GPU":
        import intel_extension_for_pytorch
        model = model.to("xpu")

    tokenizer = AutoTokenizer.from_pretrained(path_to_model, trust_remote_code=shared.args.trust_remote_code)

    return model, tokenizer


def get_max_memory_dict():
    max_memory = {}
    max_cpu_memory = shared.args.cpu_memory.strip() if shared.args.cpu_memory is not None else '99GiB'
    if shared.args.gpu_memory:
        memory_map = list(map(lambda x: x.strip(), shared.args.gpu_memory))
        for i in range(len(memory_map)):
            max_memory[i] = f'{memory_map[i]}GiB' if not re.match('.*ib$', memory_map[i].lower()) else memory_map[i]

        max_memory['cpu'] = f'{max_cpu_memory}GiB' if not re.match('.*ib$', max_cpu_memory.lower()) else max_cpu_memory

    # If --auto-devices is provided standalone, try to get a reasonable value
    # for the maximum memory of device :0
    elif shared.args.auto_devices:
        if is_xpu_available():
            total_mem = (torch.xpu.get_device_properties(0).total_memory / (1024 * 1024))
        else:
            total_mem = (torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))

        suggestion = round((total_mem - 1000) / 1000) * 1000
        if total_mem - suggestion < 800:
            suggestion -= 1000

        suggestion = int(round(suggestion / 1000))
        logger.warning(f"Auto-assiging --gpu-memory {suggestion} for your GPU to try to prevent out-of-memory errors. You can manually set other values.")
        max_memory[0] = f'{suggestion}GiB'
        max_memory['cpu'] = f'{max_cpu_memory}GiB' if not re.match('.*ib$', max_cpu_memory.lower()) else max_cpu_memory

    return max_memory if len(max_memory) > 0 else None


def clear_torch_cache():
    gc.collect()
    if not shared.args.cpu or shared.args.device == "GPU":
        if is_xpu_available():
            torch.xpu.empty_cache()


def unload_model():
    shared.model = shared.tokenizer = None
    shared.model_name = 'None'
    shared.model_dirty_from_training = False
    clear_torch_cache()


def reload_model():
    unload_model()
    shared.model, shared.tokenizer = load_model(shared.model_name)
