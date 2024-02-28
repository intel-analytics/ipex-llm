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

import torch
import intel_extension_for_pytorch as ipex
import time
import argparse

from transformers import AutoModel, AutoTokenizer
from bigdl.llm import optimize_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict Tokens using `chat()` API for VisualGLM model"
    )
    parser.add_argument(
        "--repo-id-or-model-path",
        type=str,
        default="THUDM/visualglm-6b",
        help="The huggingface repo id for the VisualGLM (e.g. `THUDM/visualglm-6b`) to be downloaded"
        ", or the path to the huggingface checkpoint folder",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Image path for the input image that the chat will focus on",
    )
    parser.add_argument(
        "--n-predict", type=int, default=512, help="Max tokens to predict"
    )

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    image_path = args.image_path

    # Load model
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype="auto", low_cpu_mem_usage=True
    ).half()

    # With only one line to enable BigDL-LLM optimization on model
    model = optimize_model(model)

    model = model.to("xpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    history = []
    while True:
        try:
            user_input = input("用户: ")
        except EOFError:
            user_input = ""
        if not user_input:
            print("exit...")
            break

        response, history = model.chat(
            tokenizer, image_path, user_input, history=history
        )
        print(f"VisualGLM: {response} ", end="")
        print("\n")
