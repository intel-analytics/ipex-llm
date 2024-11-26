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
# Code is adapted from https://huggingface.co/docs/diffusers/main/en/using-diffusers/inference_with_lcm_lora

import torch
from diffusers import DiffusionPipeline, LCMScheduler
from ipex_llm import optimize_model
import argparse
import time


def main(args):
    pipe = DiffusionPipeline.from_pretrained(
        args.repo_id_or_model_path,
        torch_dtype=torch.float16,
    )
    pipe = optimize_model(pipe, low_bit=None)
    pipe.to("xpu")

    # set scheduler
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # load LCM-LoRA
    pipe.load_lora_weights(args.lora_weights_path)

    generator = torch.manual_seed(42)

    with torch.inference_mode():
        # warmup
        image = pipe(
            prompt=args.prompt, num_inference_steps=args.num_steps, generator=generator, guidance_scale=1.0
        ).images[0]

        # start inference
        st = time.time()
        image = pipe(
            prompt=args.prompt, num_inference_steps=args.num_steps, generator=generator, guidance_scale=1.0
        ).images[0]   
        end = time.time()
        print(f'Inference time: {end-st} s')
        image.save(args.save_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Stable Diffusion lora-lcm")
    parser.add_argument('--repo-id-or-model-path', type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                        help='The huggingface repo id for the stable diffusion model checkpoint')
    parser.add_argument('--lora-weights-path',type=str,default="latent-consistency/lcm-lora-sdxl",
                        help='The huggingface repo id for the lcm lora sdxl checkpoint')
    parser.add_argument('--prompt', type=str, default="A lovely dog on the table, detailed, 8k",
                        help='Prompt to infer')
    parser.add_argument('--save-path',type=str,default="lcm-lora-sdxl-gpu.png",
                        help="Path to save the generated figure")
    parser.add_argument('--num-steps',type=int,default=4,
                        help="Number of inference steps")
    args = parser.parse_args()
    main(args)