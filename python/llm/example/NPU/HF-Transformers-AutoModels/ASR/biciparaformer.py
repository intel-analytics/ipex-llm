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
import time
import argparse

from ipex_llm.transformers.npu_model import AutoASR
from transformers.utils import logging

logger = logging.get_logger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict Tokens using `chat()` API for npu model"
    )
    parser.add_argument(
        "--repo-id-or-model-path",
        type=str,
        default="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    )
    parser.add_argument('--image-url-or-path', type=str,
                        default='http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg',
                        help='The URL or path to the image to infer')
    parser.add_argument('--prompt', type=str, default="What is in the image?",
                        help='Prompt to infer')
    parser.add_argument("--n-predict", type=int, default=32, help="Max tokens to predict")
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--disable-transpose-value-cache", action="store_true", default=False)
    parser.add_argument("--intra-pp", type=int, default=2)
    parser.add_argument("--inter-pp", type=int, default=2)

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    model = AutoASR.from_pretrained(
        model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        attn_implementation="eager",
        load_in_low_bit="sym_int8",
        low_cpu_mem_usage=True,
        funasr_model=True,
        optimize_model=True,
        max_output_len=args.max_output_len,
        max_prompt_len=args.max_prompt_len,
        intra_pp=args.intra_pp,
        inter_pp=args.inter_pp,
        transpose_value_cache=not args.disable_transpose_value_cache,
    )

    res = model.generate(input=f"{model.model_path}/example/asr_example.wav", 
                batch_size_s=300, 
                hotword='魔搭')
    print(res)
