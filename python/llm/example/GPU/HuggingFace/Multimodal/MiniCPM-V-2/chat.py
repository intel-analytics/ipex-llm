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



from typing import List, Tuple, Optional, Union
import math
import timm
import torch
import torch.nn.functional as F

# patched: `timm` has limited support for XPU backend, so we need to use CPU as a workaround
def resample_abs_pos_embed(
        posemb: torch.Tensor,
        new_size: List[int],
        old_size: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = 'bicubic',
        antialias: bool = True,
        verbose: bool = False,
):
    # sort out sizes, assume square if old size not provided
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb

    if old_size is None:
        hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
        old_size = hw, hw

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.float()  # interpolate needs float32
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    #posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = F.interpolate(posemb.to("cpu"), size=new_size, mode=interpolation, antialias=antialias).to(posemb.device)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    posemb = posemb.to(orig_dtype)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)

    if not torch.jit.is_scripting() and verbose:
        _logger.info(f'Resized position embedding: {old_size} to {new_size}.')

    return posemb


def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
    if self.pos_embed is None:
        return x.view(x.shape[0], -1, x.shape[-1])

    if self.dynamic_img_size:
        B, H, W, C = x.shape
        pos_embed = resample_abs_pos_embed(
            self.pos_embed,
            (H, W),
            num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
        )
        x = x.view(B, -1, C)
    else:
        pos_embed = self.pos_embed

    to_cat = []
    if self.cls_token is not None:
        to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
    if self.reg_token is not None:
        to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

    if self.no_embed_class:
        # deit-3, updated JAX (big vision)
        # position embedding does not overlap with class token, add then concat
        x = x + pos_embed
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
    else:
        # original timm, JAX, and deit vit impl
        # pos_embed has entry for class token, concat then add
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
        x = x + pos_embed

    return self.pos_drop(x)


setattr(timm.models.VisionTransformer, "_pos_embed", _pos_embed)


import os
import time
import argparse
import requests
import torch
from PIL import Image
from ipex_llm.transformers import AutoModel
from transformers import AutoTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for openbmb/MiniCPM-V-2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="openbmb/MiniCPM-V-2",
                        help='The huggingface repo id for the openbmb/MiniCPM-V-2 model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--image-url-or-path', type=str,
                        default='http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg',
                        help='The URL or path to the image to infer')
    parser.add_argument('--prompt', type=str, default="What is in the image?",
                        help='Prompt to infer')
    parser.add_argument('--stream', action='store_true',
                        help='Whether to chat in streaming mode')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    image_path = args.image_url_or_path
    
    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    # When running LLMs on Intel iGPUs for Windows users, we recommend setting `cpu_embedding=True` in the from_pretrained function.
    # This will allow the memory-intensive embedding layer to utilize the CPU instead of iGPU.
    model = AutoModel.from_pretrained(model_path, 
                                      load_in_low_bit="asym_int4",
                                      optimize_model=True,
                                      trust_remote_code=True,
                                      use_cache=True,
                                      modules_to_not_convert=["vpm", "resampler"])
    model = model.half().to('xpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    model.eval()

    query = args.prompt
    if os.path.exists(image_path):
       image = Image.open(image_path).convert('RGB')
    else:
       image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')

    # Generate predicted tokens
    # here the prompt tuning refers to https://huggingface.co/openbmb/MiniCPM-V-2/blob/main/README.md
    msgs = [{'role': 'user', 'content': args.prompt}]

    # ipex_llm model needs a warmup, then inference time can be accurate
    res, context, _ = model.chat(
     image=image,
     msgs=msgs,
     context=None,
     tokenizer=tokenizer,
     sampling=False,
     temperature=0.7
    )
    if args.stream:
        res, context, _ = model.chat(
        image=image,
        msgs=msgs,
        context=None,
        tokenizer=tokenizer,
        sampling=False,
        temperature=0.7
        )

        print('-'*20, 'Input Image', '-'*20)
        print(image_path)
        print('-'*20, 'Input Prompt', '-'*20)
        print(args.prompt)
        print('-'*20, 'Stream Chat Output', '-'*20)
        for new_text in res:
            print(new_text, flush=True, end='')
    else:
        st = time.time()
        res, context, _ = model.chat(
        image=image,
        msgs=msgs,
        context=None,
        tokenizer=tokenizer,
        sampling=False,
        temperature=0.7
        )
        torch.xpu.synchronize()
        end = time.time()

        print(f'Inference time: {end-st} s')
        print('-'*20, 'Input Image', '-'*20)
        print(image_path)
        print('-'*20, 'Input Prompt', '-'*20)
        print(args.prompt)
        print('-'*20, 'Chat Output', '-'*20)
        print(res)
