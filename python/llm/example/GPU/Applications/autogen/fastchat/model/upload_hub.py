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
Upload weights to huggingface.

Usage:
python3 -m fastchat.model.upload_hub --model-path ~/model_weights/vicuna-13b --hub-repo-id lmsys/vicuna-13b-v1.3
"""
import argparse
import tempfile

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def upload_hub(model_path, hub_repo_id, component, private):
    if component == "all":
        components = ["model", "tokenizer"]
    else:
        components = [component]

    kwargs = {"push_to_hub": True, "repo_id": hub_repo_id, "private": args.private}

    if "model" in components:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        with tempfile.TemporaryDirectory() as tmp_path:
            model.save_pretrained(tmp_path, **kwargs)

    if "tokenizer" in components:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        with tempfile.TemporaryDirectory() as tmp_path:
            tokenizer.save_pretrained(tmp_path, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--hub-repo-id", type=str, required=True)
    parser.add_argument(
        "--component", type=str, choices=["all", "model", "tokenizer"], default="all"
    )
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    upload_hub(args.model_path, args.hub_repo_id, args.component, args.private)
