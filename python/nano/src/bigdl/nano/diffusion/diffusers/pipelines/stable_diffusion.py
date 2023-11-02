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
import functools
import diffusers

from huggingface_hub import snapshot_download
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline
)
from diffusers import AutoencoderKL, UNet2DConditionModel

from bigdl.nano.diffusion.utils.paths import *
from bigdl.nano.pytorch import InferenceOptimizer
from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.diffusion.common.optimize import load_optimized_unet, load_optimized_vae_decoder


def inference_autocast(forward):
    """Inference autocast."""

    @functools.wraps(forward)
    def wrapper(self, *args, **kwargs):
        # TODO handle cases if unet/vae has different precision
        with InferenceOptimizer.get_context(self.unet):  # TODO
            output = forward(self, *args, **kwargs)  # Call the original function
        return output
    return wrapper


scheduler_map = dict(
    ddim=diffusers.DDIMScheduler,
    dpmpp_2=diffusers.DPMSolverMultistepScheduler,
    k_dpm_2=diffusers.KDPM2DiscreteScheduler,
    k_dpm_2_a=diffusers.KDPM2AncestralDiscreteScheduler,
    k_dpmpp_2=diffusers.DPMSolverMultistepScheduler,
    k_euler=diffusers.EulerDiscreteScheduler,
    k_euler_a=diffusers.EulerAncestralDiscreteScheduler,
    k_heun=diffusers.HeunDiscreteScheduler,
    k_lms=diffusers.LMSDiscreteScheduler,
    plms=diffusers.PNDMScheduler,
)
default_scheduler = 'k_lms'


def _preload_ov(pretrained_model_name_or_path, device, precision):
    from bigdl.nano.diffusion.diffusers.modules import NanoUNet
    unet = load_optimized_unet(
        cache_dir=os.path.join(pretrained_model_name_or_path, 'unet'),
        device=device, precision=precision)
    wrapped_unet = NanoUNet(unet)
    vae_decoder = load_optimized_vae_decoder(
        cache_dir=os.path.join(pretrained_model_name_or_path, 'vae'),
        device=device, precision=precision)
    return wrapped_unet, vae_decoder


def _postload_ov(pipe, unet, value_decoder):
    # load unet and vae
    pipe.unet = unet
    pipe.vae.decoder = value_decoder


def _load_ipex(pipe, device, precision):
    if precision == 'float32':
        vae = InferenceOptimizer.trace(
            pipe.vae, device=device,
            use_ipex=True, channels_last=True)
        text_encoder = InferenceOptimizer.trace(
            pipe.text_encoder, device=device,
            use_ipex=True, channels_last=True)
        unet = InferenceOptimizer.trace(
            pipe.unet, device=device,
            use_ipex=True, channels_last=True)
    elif precision == 'bfloat16':
        vae = InferenceOptimizer.quantize(
            pipe.vae, device=device, use_ipex=True,
            precision="bf16")
        text_encoder = InferenceOptimizer.quantize(
            pipe.text_encoder, device=device,
            use_ipex=True, precision="bf16")
        unet = InferenceOptimizer.quantize(
            pipe.unet, device=device,
            use_ipex=True, precision="bf16")
    elif precision == 'float16':
        vae = InferenceOptimizer.quantize(
            pipe.vae, device=device,
            use_ipex=True, channels_last=True,
            precision="fp16")
        text_encoder = InferenceOptimizer.quantize(
            pipe.text_encoder, device=device,
            use_ipex=True, channels_last=True,
            precision="fp16")
        unet = InferenceOptimizer.quantize(
            pipe.unet, device=device,
            use_ipex=True, channels_last=True,
            precision="fp16")
    else:
        raise ValueError(f'Unsupported precision {precision}, \
                         available options are: {["float32", "bfloat16", "float16"]}')
    pipe.vae = vae
    pipe.text_encoder = text_encoder
    pipe.unet = unet


class NanoDiffusionPipeline:
    """NanoDiffusionPipeline."""

    @staticmethod
    def from_pretrained(
            cls: DiffusionPipeline,
            pretrained_model_name_or_path,
            scheduler=default_scheduler,
            device='CPU',
            precision='bfloat16',
            backend='IPEX',
            **kwargs,):
        """Get the pretrained model."""
        scheduler = scheduler_map[scheduler].from_pretrained(
            pretrained_model_name_or_path,
            subfolder='scheduler')
        # load pipeline
        if backend.lower() == 'ov':
            if not os.path.exists(pretrained_model_name_or_path):
                try:
                    pretrained_model_name_or_path = snapshot_download(pretrained_model_name_or_path)
                except Exception:
                    msg = "Could not find local `model_path` in given local path, \
                                          could not find the model on Huggingface Hub \
                                          by given `model_path` as repo_id either."

                    invalidInputError(False, errMsg=msg)
            ov_unet, ov_vae_decoder = _preload_ov(pretrained_model_name_or_path, device, precision)

            text_encoder = CLIPTextModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder='text_encoder')
            tokenizer = CLIPTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                subfolder='tokenizer')
            safety_checker = kwargs.get('safety_checker', None)
            feature_extractor = CLIPFeatureExtractor.from_pretrained(
                pretrained_model_name_or_path,
                subfolder='feature_extractor')
            requires_safety_checker = kwargs.get('requires_safety_checker', False)
            # do not load unet/vae to save RAM
            dummy_unet = UNet2DConditionModel()
            # TODO create a new config without preloading ov for the next line
            dummy_unet._internal_dict = ov_unet.config  # TODO
            dummy_vae = AutoencoderKL.from_pretrained(
                pretrained_model_name_or_path,
                subfolder='vae')
            pipe = cls(
                vae=dummy_vae, text_encoder=text_encoder,
                tokenizer=tokenizer, unet=dummy_unet, scheduler=scheduler,
                safety_checker=safety_checker, feature_extractor=feature_extractor,
                requires_safety_checker=requires_safety_checker)

            _postload_ov(pipe, ov_unet, ov_vae_decoder)

        elif backend.lower() == 'ipex':
            pipe = cls.from_pretrained(pretrained_model_name_or_path, scheduler=scheduler, **kwargs)

            _load_ipex(pipe, device, precision)
        else:
            invalidInputError(False, errMsg=f'Backend{backend} not supported.')

        return pipe


class NanoStableDiffusionPipeline(StableDiffusionPipeline):
    """NanoStableDiffusionPipeline."""

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Get the pretrained model."""
        base = NanoDiffusionPipeline.from_pretrained(
            StableDiffusionPipeline,
            *args, **kwargs)
        return cls(
            vae=base.vae, text_encoder=base.text_encoder,
            tokenizer=base.tokenizer, unet=base.unet,
            scheduler=base.scheduler, safety_checker=base.safety_checker,
            feature_extractor=base.feature_extractor,
            requires_safety_checker=base.requires_safety_checker)

    @inference_autocast
    def __call__(self, *args, **kwargs):
        """Inference."""
        return super().__call__(*args, **kwargs)


class NanoStableDiffusionImg2ImgPipeline(StableDiffusionImg2ImgPipeline):
    """NanoStableDiffusionImg2ImgPipeline."""

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Get the pretrained model."""
        base = NanoDiffusionPipeline.from_pretrained(
            StableDiffusionImg2ImgPipeline,
            *args, **kwargs)
        return cls(
            vae=base.vae, text_encoder=base.text_encoder,
            tokenizer=base.tokenizer, unet=base.unet, scheduler=base.scheduler,
            safety_checker=base.safety_checker, feature_extractor=base.feature_extractor,
            requires_safety_checker=base.requires_safety_checker)

    @inference_autocast
    def __call__(self, *args, **kwargs):
        """Inference."""
        return super().__call__(*args, **kwargs)
