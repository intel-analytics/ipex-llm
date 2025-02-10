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
import argparse
from ipex_llm.transformers.npu_model import AutoModelForCausalLM
import transformers
from transformers import AutoTokenizer
from transformers.utils import logging
from packaging import version
import os
import shutil
import time


logger = logging.get_logger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert LLM for C++ NPU inference and save"
    )
    parser.add_argument(
        "--repo-id-or-model-path",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="The huggingface repo id for the model to be downloaded"
        ", or the path to the huggingface checkpoint folder.",
    )
    parser.add_argument("--save-directory", type=str,
        required=True,
        help="The path of folder to save converted model, "
            "If path not exists, lowbit model will be saved there. "
            "Else, program will raise error.",
    )
    parser.add_argument("--max-context-len", type=int, default=1024)
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--quantization-group-size", type=int, default=0)
    parser.add_argument('--low-bit', type=str, default="sym_int4",
                        help='Low bit optimizations that will be applied to the model.')
    parser.add_argument("--keep-ir", action="store_true")
    parser.add_argument("--disable-compile-blob", action="store_true") 

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    save_dir = args.save_directory
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    trans_version = transformers.__version__
    if version.parse(trans_version) >= version.parse("4.45.0"):
        tokenizer_json = os.path.join(model_path, "tokenizer.json")
        dst_path = os.path.join(save_dir, "tokenizer.json")
        shutil.copy(tokenizer_json, dst_path)
    else:
        tokenizer.save_pretrained(save_dir)

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 optimize_model=True,
                                                 load_in_low_bit=args.low_bit,
                                                 max_context_len=args.max_context_len,
                                                 max_prompt_len=args.max_prompt_len,
                                                 quantization_group_size=args.quantization_group_size,
                                                 torch_dtype=torch.float16,
                                                 attn_implementation="eager",
                                                 trust_remote_code=True,
                                                 save_directory=save_dir,
                                                 keep_ir=args.keep_ir,
                                                 compile_blob=not args.disable_compile_blob)
    t1 = time.perf_counter()


    print("-" * 80)
    print(f"Convert model cost {t1 - t0}s.")
    print(f"finish save model to {save_dir}")
    print("success shut down")
