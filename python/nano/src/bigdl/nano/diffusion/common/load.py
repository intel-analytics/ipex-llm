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
import time
from collections import OrderedDict
from huggingface_hub import snapshot_download

from bigdl.nano.pytorch import InferenceOptimizer


def _get_cache_path(base_dir, accelerator="jit",
                    ipex=True, precision='float32',
                    low_memory=False,
                    additional_suffix=None,
                    lora_name=None):
    # base_dir = os.path.join(base_dir, f"unet")
    model_dir = [precision]
    if accelerator:
        model_dir.append(accelerator)
    if ipex and accelerator != "openvino":
        model_dir.append("ipex")
        if low_memory:
            model_dir.append('low_memory')
    if lora_name is not None:
        lora_name = lora_name.replace("/", "-")
        model_dir.append(lora_name)
    # if device != 'CPU':
    #     model_dir.append(device)
    if additional_suffix is not None:
        model_dir.append(additional_suffix)
    model_dir = "_".join(model_dir)
    return os.path.join(base_dir, model_dir)


def try_load_existing_model(attributes, cache_dir,
                            accelerator, ipex, precision,
                            low_memory, device,
                            additional_suffix=None,
                            lora_name=None, return_model=True):
    cache_path = _get_cache_path(cache_dir, accelerator=accelerator,
                                 ipex=ipex, precision=precision, low_memory=low_memory,
                                 additional_suffix=additional_suffix, lora_name=lora_name)
    if os.path.exists(cache_path):
        try:
            if return_model:
                print(f"Loading the existing cache from {cache_path} to {device}")
                nano_model = InferenceOptimizer.load(cache_path, device=device)
                for k, v in attributes.items():
                    if k not in dir(nano_model):
                        setattr(nano_model, k, v)
                return (nano_model, cache_path)
            else:
                print(f"The existing cache {cache_path} exists, don't load it")
                return ("exists", cache_path)  # placeholder
        except Exception as e:
            print(f"The cache path {cache_path} exists, \
                  but failed to load. Error message: {str(e)}")
    return (None, cache_path)


def load_optimized_unet(
        cache_dir=None,
        unet_attributes=None,
        accelerator="openvino",
        ipex=True,
        precision='float32',
        device='CPU',
        low_memory=False,
        lora_name=None,
        additional_suffix=None):
    t_start = time.perf_counter()
    if cache_dir is None and unet_attributes is None:
        print(f"You should provide either unet_attributes or cache_dir.")
        return None
    if cache_dir is None:
        cache_dir = unet_attributes["config"]._name_or_path
    if unet_attributes is None:
        attr_file_path = os.path.join(cache_dir, "attrs.pkl")
        if os.path.exists(attr_file_path):
            import pickle
            with open(attr_file_path, "rb") as f:
                unet_attributes = pickle.load(f)
        elif os.path.exists(os.path.join(cache_dir, "config.json")):
            import json
            from diffusers.configuration_utils import FrozenDict
            with open(os.path.join(cache_dir, "config.json")) as f:
                unet_attributes = {}
                conf = json.load(f)
                conf_obj = FrozenDict(conf)
                unet_attributes["config"] = conf_obj
                unet_attributes["conv_in_in_channels"] = unet_attributes["config"].in_channels
                unet_attributes["in_channels"] = unet_attributes["config"].in_channels
        else:
            print(f"Cannot find the unet attributes file or config.json, \
                  please provide unet_attributes or config.json.")
    conv_in = OrderedDict()
    conv_in.in_channels = unet_attributes["conv_in_in_channels"]

    nano_unet, expect_path = try_load_existing_model(
        unet_attributes, cache_dir,
        accelerator=accelerator,
        ipex=ipex, precision=precision,
        low_memory=low_memory, device=device,
        additional_suffix=additional_suffix,
        lora_name=lora_name)
    t_end = time.perf_counter()
    if nano_unet is None:
        raise Exception(f"You have to download the \
                        optimized nano unet models. \
                        Expected path: {expect_path}")
    else:
        print(f"Load unet in {t_end - t_start}s")
        setattr(nano_unet, "conv_in", conv_in)
        setattr(nano_unet.config, "_name_or_path", cache_dir)
    return nano_unet


def load_optimized_vae_decoder(
        cache_dir,
        accelerator="openvino",
        ipex=True,
        precision='float32',
        device='CPU',
        low_memory=False,):
    t_start = time.perf_counter()
    decoder_path = os.path.join(cache_dir, "decoder")
    nano_vae_decoder, cache_dir = try_load_existing_model(
        {}, decoder_path, accelerator=accelerator,
        ipex=ipex, precision=precision,
        low_memory=low_memory, device=device)

    t_end = time.perf_counter()

    if nano_vae_decoder is None:
        raise Exception(f"You have to download the \
                        optimized nano vae decoder models. \
                        Expected path: {cache_dir}")
    else:
        print(f"Load vae decoder in {t_end - t_start}s")
    return nano_vae_decoder


def load_optimized_controlnet(
        controlnet=None,
        model_id=None,
        local_controlnet_dir=None,
        accelerator="jit",
        ipex=True,
        precision='float32',
        device='CPU',
        low_memory=False,):
    t_start = time.perf_counter()
    # TODO: here just got model id, not the model dir, need a better way to obtain dir
    if controlnet is not None:
        name_or_path = controlnet._name_or_path
    else:
        name_or_path = model_id
    if not os.path.exists(name_or_path):
        name_or_path = snapshot_download(
            name_or_path,
            cache_dir=local_controlnet_dir,
            local_files_only=True)
    nano_controlnet, cache_path = try_load_existing_model(
        {}, name_or_path, accelerator=accelerator,
        ipex=ipex, precision=precision,
        low_memory=low_memory, device=device)
    t_end = time.perf_counter()
    if nano_controlnet is None:
        raise Exception(f"You have to download the\
                        optimized nano unet models. Expected path: {name_or_path}")
    print(f"Load controlnet from {cache_path} in {t_end - t_start}s")
    return nano_controlnet
