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
        description="Transcribe speech to text using `generate()` API for npu model"
    )
    parser.add_argument(
        "--repo-id-or-model-path",
        type=str,
        default="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    )
    parser.add_argument('--load_in_low_bit', type=str, default="sym_int8",
                        help='Load in low bit to use')
    parser.add_argument("--intra-pp", type=int, default=2)
    parser.add_argument("--inter-pp", type=int, default=2)

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    model = AutoASR.from_pretrained(
        model=model_path,
        attn_implementation="eager",
        load_in_low_bit=args.load_in_low_bit,
        low_cpu_mem_usage=True,
        funasr_model=True,
        optimize_model=True,
        intra_pp=args.intra_pp,
        inter_pp=args.inter_pp,
    )

    res = model.generate(input=f"{model.model_path}/example/asr_example.wav",
                         batch_size_s=300,
                         hotword='魔搭')
    print(res)
