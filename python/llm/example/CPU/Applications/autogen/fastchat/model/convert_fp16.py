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
# Copyright 2023 The FastChat team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""
Usage:
python3 -m fastchat.model.convert_fp16 --in in-folder --out out-folder
"""
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def convert_fp16(in_checkpoint, out_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(in_checkpoint, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        in_checkpoint, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.save_pretrained(out_checkpoint)
    tokenizer.save_pretrained(out_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-checkpoint", type=str, help="Path to the model")
    parser.add_argument("--out-checkpoint", type=str, help="Path to the output model")
    args = parser.parse_args()

    convert_fp16(args.in_checkpoint, args.out_checkpoint)
