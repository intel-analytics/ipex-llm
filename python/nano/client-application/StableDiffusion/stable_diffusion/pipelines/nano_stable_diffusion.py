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
import inspect
import numpy as np
from tqdm.auto import tqdm
import multiprocessing as mp
from bigdl.nano.pytorch import InferenceOptimizer
from typing import Callable, List, Optional, Union
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers.schedulers import *
from diffusers.utils import logging
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.models import UNet2DConditionModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

scheduler_map = {
    "DDIM": DDIMScheduler,
    "Euler A": EulerAncestralDiscreteScheduler,
    "Euler": EulerDiscreteScheduler,
    "LMD": LMSDiscreteScheduler,
    "PNDMS": PNDMScheduler,
    "DPM-Solver": DPMSolverMultistepScheduler
}


class NanoStableDiffusionPipeline:
    def __init__(self, stable_diffusion_pipeline=None, num_workers=1):
        if stable_diffusion_pipeline is not None:
            self.device = stable_diffusion_pipeline.device
            self.vae = stable_diffusion_pipeline.vae
            self.unet = stable_diffusion_pipeline.unet
            self.tokenizer = stable_diffusion_pipeline.tokenizer
            self.text_encoder = stable_diffusion_pipeline.text_encoder
            self.scheduler = stable_diffusion_pipeline.scheduler
            self.safety_checker = stable_diffusion_pipeline.safety_checker
            self.feature_extractor = stable_diffusion_pipeline.feature_extractor
        self.num_workers = num_workers

    @classmethod
    @torch.no_grad()
    def from_pretrained(cls, pretrained_model_path, **kwargs):
        pipe = cls()
        pipe.device = torch.device('cpu')
        pipe.vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder='vae')
        vae_decoder_path = pipe._get_cache_path(pretrained_model_path, vae=True, **kwargs)
        if vae_decoder_path:
            print(f"Loading existing optimized vae decoder from {vae_decoder_path}...")
            decoder = InferenceOptimizer.load(vae_decoder_path, device=kwargs['device'])
            setattr(pipe.vae, "decoder", decoder)
        cache_path = pipe._get_cache_path(pretrained_model_path, **kwargs) # get model path by optimization pipeline(accelerator), precision
        if "int8" in cache_path:
            # load inc model needs original model
            unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder='unet')
            pipe.unet = InferenceOptimizer.load(cache_path, model=unet)
        else:
            pipe.unet = InferenceOptimizer.load(cache_path, device=kwargs['device'])
        setattr(pipe.unet, "in_channels", 4)
        pipe.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path + '/tokenizer')
        pipe.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path + '/text_encoder')
        pipe.safety_checker = None
        pipe.feature_extractor = None
        pipe.scheduler = None
        return pipe

    def switch_scheduler(self, scheduler_name, local_scheduler_path=None, model_id="CompVis/stable-diffusion-v1-4"):
        if scheduler_name in scheduler_map:
            scheduler_cls = scheduler_map[scheduler_name]
            scheduler = None
            if local_scheduler_path is not None:
                if os.path.isdir(local_scheduler_path):
                    scheduler = scheduler_cls.from_pretrained(local_scheduler_path)
            
            if scheduler is None:
                scheduler = scheduler_cls.from_pretrained(model_id, subfolder="scheduler")
                
            self.scheduler = scheduler
        else:
            raise ValueError(f"Only support scheduler names {str(list(scheduler_map.keys()))}, but got {scheduler_name}")

    @torch.no_grad()
    def convert_pipeline(
        self, 
        accelerator="jit", 
        ipex=True, 
        precision='float32',
        device='CPU',
        samples=None, 
        height=512,
        width=512,
        low_memory=False,
        cache=False, 
        cache_dir=None,
        fail_if_no_cache=False, 
        channels_last=False,
        num_inference_steps=50,
        accuracy_drop=3000):
        """
        Trace a torch.nn.Module and convert it into an accelerated module for inference.

        For example, this function returns a PytorchOpenVINOModel when accelerator=='openvino'.

        :param low_memory: only valid when accelerator="jit" and ipex=True, model will use less memory during inference
        :cache_dir: the directory to save the converted model
        """
        generator = torch.Generator(device="cpu")
        generator.manual_seed(1)
        loaded = False

        latent_shape = (2, self.unet.in_channels, height // 8, width // 8)
        image_latents = torch.randn(latent_shape, generator = generator, device = "cpu", dtype=torch.float32)
        encoder_hidden_states = self.get_encoder_hidden_states()
        input_sample = (image_latents, torch.Tensor([980]).long(), encoder_hidden_states)

        # vae decoder
        if accelerator == "openvino" and precision == "float16" and device == "GPU":
            print(f"Start optimizing vae decoder...")
            decoder_loaded = False
            if cache:
                assert cache_dir is not None, f"Please provide cache_dir if cache=True."
                vae_cache_path = self._get_cache_path(cache_dir, accelerator=accelerator, precision=precision, device=device, vae=True)
                if vae_cache_path and os.path.exists(vae_cache_path):
                    try:
                        print(f"Loading the existing cache from {vae_cache_path}")
                        nano_vae_decoder = InferenceOptimizer.load(vae_cache_path, device=device)
                        decoder_loaded = True
                    except Exception as e:
                        decoder_loaded = False
                        print(f"The cache path {vae_cache_path} exists, but failed to load. Error message: {str(e)}")
            if not decoder_loaded:
                vae_decoder = self.vae.decoder
                nano_vae_decoder = InferenceOptimizer.quantize(vae_decoder, 
                                                               accelerator=accelerator,
                                                               input_sample=torch.randn((1, self.unet.in_channels, height // 8, width // 8), 
                                                                                        generator=generator, 
                                                                                        device="cpu", 
                                                                                        dtype=torch.float32), 
                                                               precision="fp16", 
                                                               dynamic_axes=False)
            setattr(self.vae, "decoder", nano_vae_decoder)
            if cache:
                logger.info(f"Caching the converted vae decoder model to {vae_cache_path}")
                InferenceOptimizer.save(nano_vae_decoder, vae_cache_path)

        print(f"Start optimizing unet...")
        unet_input_names = ["sample", "timestep", "encoder_hidden_states"]
        unet_output_names = ["unet_output"]
        unet_dynamic_axes = {"sample": [0], "encoder_hidden_states": [0], "unet_output": [0]}

        if cache:
            assert cache_dir is not None, f"Please provide cache_dir if cache=True."
            cache_path = self._get_cache_path(cache_dir, accelerator=accelerator, ipex=ipex, precision=precision, low_memory=low_memory, device=device)
            if precision == "bfloat16" and accelerator != "openvino":
                pass
            elif os.path.exists(cache_path):
                try:
                    print(f"Loading the existing cache from {cache_path}")
                    nano_unet = InferenceOptimizer.load(cache_path, device=device)
                    loaded = True
                except Exception as e:
                    loaded = False
                    print(f"The cache path {cache_path} exists, but failed to load. Error message: {str(e)}")


        print("precision is", precision)
        if not loaded:
            if fail_if_no_cache:
                raise Exception("You have to download the model to nano_stable_diffusion folder")

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
                        raise ValueError("IPEX should be True if accelerator is None and precision is float32.")
                elif accelerator == "openvino":
                    extra_args["input_names"] = unet_input_names
                    extra_args["output_names"] = unet_output_names
                    # Nano will deal with the GPU/VPU dynamic axes issue
                    extra_args["dynamic_axes"] = unet_dynamic_axes
                    extra_args["device"] = device
                else:
                    raise ValueError(f"The accelerator can be one of `None`, `jit`, and `openvino` if the precision is float32, but got {accelerator}")
                nano_unet = InferenceOptimizer.trace(self.unet,
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
                    # minimal UI integrates FP16 CPU and GPU options because there two models are the same
                    # so we hardcode here to avoid GPU not available issue
                    extra_args["device"] = 'CPU'
                    extra_args["input_names"] = unet_input_names
                    extra_args["output_names"] = unet_output_names
                    # Nano will deal with the GPU/VPU dynamic axes issue
                    extra_args["dynamic_axes"] = False

                    if precision_short == "int8":
                        # TODO: openvino int8 here
                        raise ValueError("OpenVINO int8 quantization is not supported.")
                elif accelerator == "onnxruntime":
                    raise ValueError(f"Onnxruntime {precision_short} quantization is not supported.")
                else:
                    # PyTorch bf16
                    if precision_short == "bf16":
                        # Ignore jit & ipex
                        if accelerator == "jit":
                            raise ValueError(f"JIT {precision_short} quantization is not supported.")
                        extra_args["channels_last"] = channels_last
                    elif precision_short == "int8":
                        # INC int8 or INC ipex int8
                        if samples is not None:
                            # get real calibration sample
                            input_sample = samples[0][0]

                        class CalibDataLoader(object):
                            def __init__(self, samples):
                                self.batch_size = 1
                                self.data = samples
                                self.len = len(samples)

                            def __iter__(self):
                                for i in range(self.len):
                                    data = self.data[i]
                                    input, output = data
                                    yield input, output

                        prompt = "a photo of an astronaut riding a horse on mars"
                        generator_eval = torch.Generator("cpu").manual_seed(77)
                        eval_image = self(prompt, generator=generator_eval, num_inference_steps=num_inference_steps)[0]
                        eval_image.save("eval_image.jpg")

                        def eval_func(model):
                            setattr(model, "in_channels", 4)
                            setattr(self, "unet", model)
                            with torch.no_grad():
                                loss = torch.nn.MSELoss()
                                generator_eval = torch.Generator("cpu").manual_seed(77)
                                new_image = self(prompt, guidance_scale=7.5, num_inference_steps=num_inference_steps,
                                                generator=generator_eval)[0]
                                new_image.save("new_image.jpg")
                                mse_score = 0
                                new = torch.from_numpy(np.array(new_image))
                                old = torch.from_numpy(np.array(eval_image))
                                new = new.to(dtype=torch.float32)
                                old = old.to(dtype=torch.float32)
                                mse_score += loss(new, old)
                                mse_score = mse_score.item()
                                return mse_score

                        if samples is None or len(samples) < 1:
                            raise ValueError("Calibration samples can't be None or empty for quantization.")

                        dataloader = CalibDataLoader(samples)
                        dataloader.collate_fn = None
                        extra_args["calib_dataloader"] = dataloader
                        extra_args["eval_func"] = eval_func
                        extra_args["accuracy_criterion"] = {"absolute": accuracy_drop,
                                                            "higher_is_better": False}
                        extra_args["max_trials"] = 10
                        extra_args["timeout"] = 0
                        extra_args["tuning_strategy"] = 'basic'
                    else:
                        raise ValueError(f"PyTorch {precision_short} quantization is not supported.")

                # unet
                nano_unet = InferenceOptimizer.quantize(self.unet,
                                                        accelerator=accelerator,
                                                        precision=precision_short,
                                                        input_sample=input_sample,
                                                        **extra_args)

            # Save model if cache=True
            if cache:
                logger.info(f"Caching the converted unet model to {cache_path}")
                InferenceOptimizer.save(nano_unet, cache_path)

        setattr(nano_unet, "in_channels", 4)
        self.unet = nano_unet
        return self
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        guidance_threshold: int = 51, 
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        return_samples: bool = False,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (
            not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
            removed_text = self.tokenizer.batch_decode(
                text_input_ids[:, self.tokenizer.model_max_length:])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""]
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(batch_size, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len,
                                                       -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # get the initial random noise unless the user supplied it

        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents_shape = (
        batch_size * num_images_per_prompt, self.unet.in_channels, height // 8, width // 8)
        latents_dtype = text_embeddings.dtype
        if latents is None:
            if self.device.type == "mps":
                # randn does not work reproducibly on mps
                latents = torch.randn(latents_shape, generator=generator, device="cpu",
                                      dtype=latents_dtype).to(
                    self.device
                )
            else:
                latents = torch.randn(latents_shape, generator=generator, device=self.device,
                                      dtype=latents_dtype)
        else:
            if latents.shape != latents_shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
            latents = latents.to(self.device)

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator

        input_samples = []
        
        for i, t in enumerate(tqdm(timesteps_tensor)):			
            if do_classifier_free_guidance and i >= guidance_threshold:
                do_classifier_free_guidance_ = False
                text_embeddings_ = text_embeddings_[:1]
            else:
                text_embeddings_ = text_embeddings
                do_classifier_free_guidance_ = do_classifier_free_guidance

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat(
                [latents] * 2) if do_classifier_free_guidance_ else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            t = t[None]
            # predict the noise residual

            noise_pred = self.unet(latent_model_input, t, text_embeddings_)
            if hasattr(noise_pred, "sample"):
                noise_pred = noise_pred.sample
            if isinstance(noise_pred, tuple):
                noise_pred = noise_pred[0]
            elif isinstance(noise_pred, dict):
                noise_pred = noise_pred["sample"]

            if return_samples:
                input_samples.append([(latent_model_input, t, text_embeddings), noise_pred])

            # perform guidance
            if do_classifier_free_guidance_:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(DiffusionPipeline.numpy_to_pil(image),
                                                          return_tensors="pt").to(
                self.device
            )
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(text_embeddings.dtype)
            )
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            image = DiffusionPipeline.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        if return_samples:
            return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept), input_samples
        else:
            return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def _get_cache_path(self, base_dir, accelerator="jit", ipex=True, precision='float32', low_memory=False, device='CPU', vae=False):
        # base_dir = os.path.expanduser(
        # 	os.getenv(
        # 		"NANO_HOME", 
        # 		os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), f"nano_stable_diffusion/nano_{accelerator}/unet")))
        model_name = "vae" if vae else "unet"
        if vae:
            if accelerator != "openvino" or precision != "float16" or device != "GPU":
                return False
        base_dir = os.path.join(base_dir, model_name)
        model_dir = [precision]
        if accelerator:
            model_dir.append(accelerator)
        if ipex and accelerator != "openvino":
            model_dir.append("ipex")
            if low_memory:
                model_dir.append('low_memory')
        # if device != 'CPU':
        #     model_dir.append(device)
        model_dir = "_".join(model_dir)
        return os.path.join(base_dir, model_dir)

    def __call__(self, prompt, **kwargs):
        imgs = self.generate(prompt=prompt, **kwargs)
        if hasattr(imgs, "images"):
            imgs = imgs.images
        return imgs

    def get_openvino_config(self, precision):
        config = {}
        if precision == "float32":
            config["INFERENCE_PRECISION_HINT"] = "f32"

        if self.num_workers > 1:
            # default use all physical cores
            import psutil
            core_num = psutil.cpu_count(logical=False)
            core_num = int(core_num / self.num_workers)
            config["CPU_THREADS_NUM"] = str(core_num)
            config["CPU_BIND_THREAD"] = "NO"
        return config

    def get_encoder_hidden_states(self, prompt=None, cfg=True):
        if prompt is None:
            prompt = [""]
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="max_length", return_tensors="pt").input_ids

        text_embeddings = self.text_encoder(
            text_input_ids.to(self.device),
            attention_mask=None,
        )
        text_embeddings = text_embeddings[0]

        # get unconditional embeddings for classifier free guidance
        if cfg:
            uncond_tokens = [""]

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device),
                attention_mask=None,
            )
            uncond_embeddings = uncond_embeddings[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings
