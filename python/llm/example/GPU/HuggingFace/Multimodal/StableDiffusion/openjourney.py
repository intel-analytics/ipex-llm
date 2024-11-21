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
# Code is adapted from https://huggingface.co/prompthero/openjourney

from diffusers import StableDiffusionPipeline
import torch
from ipex_llm import optimize_model
import argparse
import time


def main(args):
    pipe = StableDiffusionPipeline.from_pretrained(
        args.repo_id_or_model_path, 
        torch_dtype=torch.float16, 
        use_safetensors=True)
    pipe = optimize_model(pipe, low_bit=None)
    pipe = pipe.to("xpu")

    with torch.inference_mode():
        # warmup
        image = pipe(args.prompt, num_inference_steps=args.num_steps).images[0]

        # start inference
        st = time.time()
        image = pipe(args.prompt, num_inference_steps=args.num_steps).images[0]
        end = time.time()
        print(f'Inference time: {end-st} s')
        image.save(args.save_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Stable Diffusion")
    parser.add_argument('--repo-id-or-model-path', type=str, default="prompthero/openjourney",
                        help='The huggingface repo id for the stable diffusion model checkpoint')
    parser.add_argument('--prompt', type=str, default="An astronaut in the forest, detailed, 8k",
                        help='Prompt to infer')
    parser.add_argument('--save-path',type=str,default="openjourney-gpu.png",
                        help="Path to save the generated figure")
    parser.add_argument('--num-steps',type=int,default=20,
                        help="Number of inference steps")
    args = parser.parse_args()
    main(args)