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
import torch
import shutil
from collections import OrderedDict
from diffusers.models import ControlNetModel
from huggingface_hub import snapshot_download

from ..common.load import *
from ..utils.paths import *
from bigdl.nano.pytorch import InferenceOptimizer
from bigdl.nano.utils.common import invalidInputError


# convert unet to optimized nano unet
def optimize_unet(
        unet,
        unet_attributes,
        accelerator="jit",
        ipex=True,
        precision='float32',
        device='CPU',
        samples=None,
        height=512,
        width=512,
        low_memory=False,
        cache=False,
        fail_if_no_cache=False,
        channels_last=False,
        cache_dir=None,
        lora_name=None,
        return_model=True):
    """
    Trace a torch.nn.Module and convert it into an accelerated module for inference.
    For example, this function returns a PytorchOpenVINOModel when accelerator=='openvino'.
    :param low_memory: only valid when accelerator="jit"
        and ipex=True,model will use less memory during inference
    :cache_dir: the directory to save the converted model
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1)

    conv_in = OrderedDict()
    conv_in.in_channels = unet.conv_in.in_channels
    if cache_dir is None:
        cache_dir = unet_attributes["config"]._name_or_path

    latent_shape = (
        2, unet.in_channels,
        height // 8, width // 8)
    image_latents = torch.randn(
        latent_shape,
        generator=generator,
        device="cpu", dtype=torch.float32)
    cross_attention_dim = unet.config.cross_attention_dim
    encoder_hidden_states = torch.randn(
        (2, 77, cross_attention_dim), generator=generator,
        device="cpu", dtype=torch.float32)

    temp = get_dummy_unet_additional_residuals()
    down_block_additional_residuals, mid_block_additional_residual = temp

    input_sample = [
        image_latents, torch.Tensor([980]).long(),
        encoder_hidden_states, None, None, None, None,
        down_block_additional_residuals,
        mid_block_additional_residual]

    unet_input_names = [
        "sample", "timestep",
        "encoder_hidden_states",
        "down_block_additional_residuals",
        "mid_block_additional_residual"]
    unet_output_names = ["unet_output"]
    # unet_dynamic_axes = {"sample": [0], "encoder_hidden_states": [0], "unet_output": [0]}
    unet_dynamic_axes = False

    if lora_name is not None:
        input_sample[6] = torch.Tensor([0.8]).float()
        unet_input_names.insert(3, "cross_attn_scale")

    nano_unet = None
    if cache:
        nano_unet, cache_path = try_load_existing_model(
            unet_attributes,
            cache_dir, accelerator=accelerator,
            ipex=ipex, precision=precision,
            low_memory=low_memory, device=device,
            lora_name=lora_name, return_model=return_model)

    if nano_unet is None:
        if fail_if_no_cache:
            raise Exception(f"`fail_if_no_cache` is set to True, \
                but failed to find the model at \
                {unet_attributes['config']._name_or_path}, \
                optimization stopped.")

        nano_unet = nano_optimize_model(
            unet, input_sample,
            input_names=unet_input_names,
            output_names=unet_output_names,
            dynamic_axes=unet_dynamic_axes,
            accelerator=accelerator,
            ipex=ipex, precision=precision,
            device=device, samples=samples,
            low_memory=low_memory,
            channels_last=channels_last)

        # Save model if cache=True
        if cache:
            import pickle
            print(f"Caching the converted unet model to {cache_path}")
            InferenceOptimizer.save(nano_unet, cache_path)
            with open(os.path.join(cache_dir, "attrs.pkl"), "wb") as f:
                pickle.dump(unet_attributes, f)

    if not isinstance(nano_unet, str):
        setattr(nano_unet, "conv_in", conv_in)
    return nano_unet


def optimize_vae(
        vae,
        unet_in_channels=4,
        accelerator="jit",
        ipex=True,
        precision='float32',
        device='CPU',
        height=512,
        width=512,
        low_memory=False,
        cache=False,
        fail_if_no_cache=False,
        channels_last=False,
        inplace=True,
        cache_dir=None,
        return_model=True):
    """
    Trace a torch.nn.Module and convert it into an accelerated module for inference.
    For example, this function returns a PytorchOpenVINOModel when accelerator=='openvino'.
    :param low_memory: only valid when accelerator="jit"
        and ipex=True, model will use less memory during inference
    :cache_dir: the directory to save the converted model
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1)

    if cache_dir is None:
        cache_dir = vae._name_or_path
    decoder_path = os.path.join(cache_dir, "decoder")
    # TODO: encoder

    latent_shape = (1, unet_in_channels, height // 8, width // 8)
    image_latents = torch.randn(
        latent_shape,
        generator=generator,
        device="cpu", dtype=torch.float32)
    input_sample = image_latents

    nano_vae_decoder = None

    if cache:
        nano_vae_decoder, decoder_cache_path = try_load_existing_model(
            {},
            decoder_path, accelerator=accelerator,
            ipex=ipex, precision=precision,
            low_memory=low_memory, device=device,
            return_model=return_model)

    if nano_vae_decoder is None:
        if fail_if_no_cache:
            raise Exception(f"`fail_if_no_cache` \
                is set to True, but failed to find the model \
                at {decoder_path}, optimization stopped.")

        nano_vae_decoder = nano_optimize_model(
            vae.decoder,
            input_sample, accelerator=accelerator, ipex=ipex, precision=precision,
            device=device, low_memory=low_memory, channels_last=channels_last)

        # Save model if cache=True
        if cache:
            print(f"Caching the converted vae decoder model to {decoder_cache_path}")
            InferenceOptimizer.save(nano_vae_decoder, decoder_cache_path)
    if not isinstance(nano_vae_decoder, str) and inplace:
        setattr(vae, "decoder", nano_vae_decoder)
        return vae
    else:
        return nano_vae_decoder


def optimize_controlnet(
        controlnet,
        accelerator="openvino",
        ipex=False,
        precision='float16',
        device='CPU',
        samples=None,
        height=512,
        width=512,
        low_memory=False,
        cache=True,
        fail_if_no_cache=False,
        channels_last=False,
        return_model=False):
    """
    Trace a torch.nn.Module and convert it into an accelerated module for inference.
    For example, this function returns a PytorchOpenVINOModel when accelerator=='openvino'.
    :param low_memory: only valid when accelerator="jit"
        and ipex=True, model will use less memory during inference
    :cache_dir: the directory to save the converted model
    """
    latent_model_input = torch.randn(2, 4, 64, 64)
    t = torch.Tensor([980]).long()
    encoder_hidden_states = torch.randn(
        2,
        77, 768)  # only work for 1.4/1.5 now, so fix it to 768
    controlnet_cond = torch.randn(2, 3, 512, 512)
    input_sample = (
        latent_model_input,
        t, encoder_hidden_states,
        controlnet_cond, None, None, None, None, False)
    controlnet_input_names = [
        "sample",
        "timestep", "encoder_hidden_states",
        "controlnet_cond"]
    controlnet_dynamic_axes = {
        "sample": [0],
        "encoder_hidden_states": [0],
        "controlnet_cond": [0]}
    name_or_path = controlnet._name_or_path
    if not os.path.exists(name_or_path):
        raise Exception(f"controlnet model does not exist \
            in {name_or_path}, optimization failed.")

    nano_controlnet = None

    if cache:
        nano_controlnet, cache_path = try_load_existing_model(
            {}, name_or_path,
            accelerator=accelerator, ipex=ipex, precision=precision,
            low_memory=low_memory, device=device, return_model=return_model)

    if nano_controlnet is None:
        if fail_if_no_cache:
            raise Exception(f"`fail_if_no_cache` is set to True, \
                            but failed to find the model at {name_or_path}, optimization stopped.")

        nano_controlnet = nano_optimize_model(
            controlnet, input_sample,
            accelerator=accelerator, ipex=ipex,
            precision=precision,
            input_names=controlnet_input_names,
            dynamic_axes=controlnet_dynamic_axes,
            device=device,
            samples=samples, low_memory=low_memory,
            channels_last=channels_last,
            mo_kwargs={
                "input_shape": "[2,4,64,64],[1],[2,77,768],[2,3,512,512]",
                "input": "sample, timestep, encoder_hidden_states, controlnet_cond"})

        # Save model if cache=True
        if cache:
            print(f"Caching the converted controlnet model to {cache_path}")
            InferenceOptimizer.save(nano_controlnet, cache_path)

    if not return_model:
        del nano_controlnet
        del controlnet
        return

    return nano_controlnet


def nano_optimize_model(
        model,
        input_sample,
        input_names=None,
        output_names=None,
        dynamic_axes=False,
        accelerator="jit",
        ipex=True,
        precision='float32',
        device='CPU',
        samples=None,
        low_memory=False,
        channels_last=False,
        mo_kwargs=None):
    extra_args = {}
    if precision == 'float32':
        if accelerator == "jit":
            weights_prepack = False if low_memory else None
            extra_args["weights_prepack"] = weights_prepack
            extra_args["use_ipex"] = ipex
            extra_args["jit_strict"] = False
            extra_args["enable_onednn"] = False
            extra_args["channels_last"] = channels_last
        elif accelerator is None:
            if ipex:
                extra_args["use_ipex"] = ipex
                extra_args["channels_last"] = channels_last
            else:
                raise ValueError("IPEX should be True if accelerator \
                    is None and precision is float32.")
        elif accelerator == "openvino":
            extra_args["input_names"] = input_names
            extra_args["output_names"] = output_names
            # # Nano will deal with the GPU/VPU dynamic axes issue
            extra_args["dynamic_axes"] = dynamic_axes
            extra_args["device"] = device
            if mo_kwargs is not None:
                extra_args.update(mo_kwargs)
        else:
            raise ValueError(f"The accelerator can be one of `None`, `jit`, \
                and `openvino` if the precision is float32, but got {accelerator}")
        optimized_model = InferenceOptimizer.trace(
            model,
            accelerator=accelerator,
            input_sample=input_sample,
            **extra_args)
    else:
        precision_map = {
            'bfloat16': 'bf16',
            'int8': 'int8',
            'float16': 'fp16'
        }
        precision_short = precision_map[precision]

        # prepare input samples, calib dataloader and eval functions
        if accelerator == "openvino":
            extra_args["device"] = device
            extra_args["input_names"] = input_names
            extra_args["output_names"] = output_names
            # Nano will deal with the GPU/VPU dynamic axes issue
            extra_args["dynamic_axes"] = dynamic_axes
            if mo_kwargs is not None:
                extra_args.update(mo_kwargs)

            if precision_short == "int8":
                # TODO: openvino int8 here
                invalidInputError(
                    precision_short != "int8",
                    errMsg="OpenVINO int8 quantization is not supported.")

        elif accelerator == "onnxruntime":
            invalidInputError(
                accelerator != "onnxruntime",
                errMsg=f"Onnxruntime {precision_short} quantization is not supported.")
        else:
            # PyTorch bf16
            if precision_short == "bf16":
                # Ignore jit & ipex
                if accelerator == "jit":
                    invalidInputError(
                        accelerator == "jit",
                        errMsg=f"JIT {precision_short} quantization is not supported.")
                extra_args["channels_last"] = channels_last
            elif precision_short == "int8":
                raise

        # unet
        optimized_model = InferenceOptimizer.quantize(
            model,
            accelerator=accelerator,
            precision=precision_short,
            input_sample=input_sample,
            **extra_args)
    return optimized_model


def save_unet_if_not_exist(unet, cache_dir=None):
    '''
    Saves an optimized UNet to local file system if it's not there
        Parameters:
            unet: the UNet2DConditionModel instance
            cache_dir: the model path to save the optimized UNet,
                if None, will use the original UNet path
    '''
    unet_attrs = unet_attributes(unet)

    # If the controlnet unet exists
    if cache_dir is None:
        cache_dir = unet_attrs["config"]._name_or_path
    exists, cache_path = try_load_existing_model(
        None, cache_dir, "openvino", ipex=False,
        precision="float16", additional_suffix="controlnet",
        return_model=False, low_memory=False, device="CPU")
    if exists:
        unet_exists, unet_cache_path = try_load_existing_model(
            None, cache_dir, "openvino", ipex=False, precision="float16",
            return_model=False, low_memory=False, device="CPU")
        if unet_exists:
            print(f"Deleting the deprecated unet...")
            shutil.rmtree(unet_cache_path)
        shutil.move(cache_path, unet_cache_path)
    else:
        opt_unet = optimize_unet(
            unet, unet_attrs, accelerator="openvino",
            ipex=False, precision="float16",
            cache=True, cache_dir=cache_dir,
            return_model=False)
        del opt_unet
    del unet


def save_vae_if_not_exist(vae, cache_dir=None):
    '''
    Saves an optimized VAE to local file system if it's not there
        Parameters:
            unet: the AutoencoderKL instance
            cache_dir: the model path to save the optimized VAE,
                    if None, will use the original VAE path
    '''
    opt_vae = optimize_vae(vae, accelerator="openvino", ipex=False,
                           precision="float16", cache=True, inplace=False,
                           cache_dir=cache_dir, return_model=False)
    del vae
    del opt_vae


def save_controlnet_if_not_exist(local_controlnet_path):
    def save_a_controlnet(repo_id):
        cache_dir = snapshot_download(repo_id, cache_dir=os.path.join(local_controlnet_path))
        controlnet = ControlNetModel.from_pretrained(cache_dir)
        print(f">> optimizing controlnet {repo_id} from {controlnet._name_or_path}")
        optimize_controlnet(controlnet,
                            accelerator="openvino",
                            ipex=False,
                            precision='float16',
                            cache=True,
                            return_model=False)
        del controlnet
    save_a_controlnet("lllyasviel/sd-controlnet-canny")
    save_a_controlnet("lllyasviel/sd-controlnet-openpose")
    save_a_controlnet("lllyasviel/sd-controlnet-depth")
    save_a_controlnet("lllyasviel/sd-controlnet-scribble")


def unet_attributes(model):
    invalidInputError(model is not None, errMsg="Please load model before saving attributes...")

    invalidInputError(
        isinstance(model, torch.nn.Module),
        errMsg="model should be a torch.nn.Module.")
    unet_attributes = {}
    for attr in dir(model):
        # if not attr.startswith('_') and not isinstance(getattr(model, attr), torch.nn.Module):
        if attr in ["config", "in_channels"]:
            unet_attributes[attr] = getattr(model, attr)
    unet_attributes["conv_in_in_channels"] = model.conv_in.in_channels
    return unet_attributes


def get_dummy_unet_additional_residuals():
    down_block_additional_residuals = []
    down_block_additional_residuals.extend([torch.zeros(2, 320, 64, 64)] * 3)
    down_block_additional_residuals.extend([torch.zeros(2, 320, 32, 32)])
    down_block_additional_residuals.extend([torch.zeros(2, 640, 32, 32)] * 2)
    down_block_additional_residuals.extend([torch.zeros(2, 640, 16, 16)])
    down_block_additional_residuals.extend([torch.zeros(2, 1280, 16, 16)] * 2)
    down_block_additional_residuals.extend([torch.zeros(2, 1280, 8, 8)] * 3)
    mid_block_additional_residual = torch.zeros(2, 1280, 8, 8)
    return down_block_additional_residuals, mid_block_additional_residual


def get_nano_cache_dir_dict(model_info, vae_repo_id=None, vae_subfolder=None):
    cache_dir_dict = {"unet": None, "vae": None}
    if model_info["format"] == "ckpt":
        from ldm.invoke.globals import Globals

        invalidInputError("weights" in model_info, errMsg="`weights` is not in model_info.")
        checkpoint_path = model_info["weights"]
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.normpath(os.path.join(Globals.root, checkpoint_path))
        cache_dir = checkpoint_path.split(".")
        cache_dir.pop()
        cache_dir = ".".join(cache_dir)
        os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
        cache_dir_dict["unet"] = os.path.join(cache_dir, "unet")
        if vae_repo_id is None:
            cache_dir_dict["vae"] = os.path.join(cache_dir, "vae")
        else:
            cache_dir_dict["vae"] = get_local_path_from_repo_id(vae_repo_id)
    else:
        if vae_repo_id is not None:
            if not os.path.isabs(vae_repo_id):
                cache_dir_dict["vae"] = get_local_path_from_repo_id(vae_repo_id)
            if vae_subfolder:
                cache_dir_dict["vae"] = os.path.join(cache_dir_dict["vae"], vae_subfolder)

    return cache_dir_dict


def get_available_devices():
    from openvino.runtime import Core
    core = Core()
    devices = core.available_devices
    avail_devices = devices
    if len(devices) == 1:
        return avail_devices
    elif len(devices) == 2:
        if devices[1] == 'GPU':
            # judge iGPU or dGPU
            device_type = core.get_property(devices[1], 'DEVICE_TYPE')
            if str(device_type) == "Type.INTEGRATED":
                avail_devices[1] = 'iGPU'
            else:
                avail_devices[1] = 'dGPU'
    elif len(devices) == 3:
        if devices[1] == 'GPU.0' and devices[2] == 'GPU.1':
            avail_devices[1] = 'iGPU'
            avail_devices[2] = 'dGPU'
    return avail_devices


def load_optimized_ov_unet(name_or_path, nano_device='iGPU', suffix=None):
    from pathlib import Path
    if name_or_path is not None:
        if not isinstance(name_or_path, Path):
            from ldm.invoke.optimize.nano_optimize import get_local_path_from_repo_id
            name_or_path = get_local_path_from_repo_id(name_or_path)
        name_or_path = os.path.join(name_or_path, "unet")
        if nano_device not in ['CPU', 'iGPU', 'dGPU']:

            invalidInputError(
                nano_device in ['CPU', 'iGPU', 'dGPU'],
                errMsg=f"Only support device `CPU`, `iGPU` and `dGPU`, but got {nano_device}")
        loaded_unet = load_optimized_unet(unet_attributes=None,
                                          accelerator="openvino",
                                          precision="float16",
                                          device=nano_device,
                                          cache_dir=name_or_path,
                                          additional_suffix=suffix)
        return loaded_unet
    else:
        return None
