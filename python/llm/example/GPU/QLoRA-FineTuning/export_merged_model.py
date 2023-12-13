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
# This file is adapted from https://github.com/tloen/alpaca-lora/blob/main/export_hf_checkpoint.py
#
# Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import torch
from transformers import LlamaTokenizer  # noqa: F402
from bigdl.llm.transformers.qlora import PeftModel, LoraConfig
from bigdl.llm.transformers import AutoModelForCausalLM
from bigdl.llm.transformers.low_bit_linear import get_block_size
import argparse
import tempfile
import shutil

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--adapter_path', type=str,)
    parser.add_argument('--output_path', type=str,)

    args = parser.parse_args()
    base_model = model_path = args.repo_id_or_model_path
    adapter_path = args.adapter_path
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    lora_config = LoraConfig.from_json_file(os.path.join(adapter_path, "adapter_config.json"))
    qa_lora = lora_config.get("qa_lora", False)

    temp_dir = None
    if qa_lora:
        # Convert the qa-lora adapter to the correct shapes
        # The default 4-bit format for qa_lora is sym_int4
        block_size = get_block_size("sym_int4")
        temp_dir = tempfile.TemporaryDirectory()
        tmpdirname = os.path.join(temp_dir.name, "adapter")
        try:
            shutil.copytree(adapter_path, tmpdirname)
        except Exception as e:
            print(f"Failed to copy adapter dir, error: {e}")
        mid_lora_path = os.path.join(tmpdirname, "adapter_model.bin")

        adapter_path = os.path.join(adapter_path, "adapter_model.bin")

        lora = torch.load(adapter_path, map_location='cpu')
        # Get lora_a names
        tmp_keys = [key for key in lora.keys() if 'lora_A' in key]

        for tmp_key in tmp_keys:
            lora_a = lora[tmp_key] / block_size
            lora[tmp_key] = torch.repeat_interleave(lora_a, block_size, dim=1)

        torch.save(lora, mid_lora_path)
        adapter_path = tmpdirname

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            # load_in_low_bit="nf4", # should load the orignal model
            torch_dtype=torch.float16,
            device_map={"": "cpu"},
        )

        lora_model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            device_map={"": "cpu"},
            torch_dtype=torch.float16,
        )

        # merge weights - new merging method from peft
        lora_model = lora_model.merge_and_unload()

        lora_model.train(False)

        lora_model_sd = lora_model.state_dict()
        deloreanized_sd = {
            k.replace("base_model.model.", ""): v
            for k, v in lora_model_sd.items()
            if "lora" not in k
        }

        base_model.save_pretrained(args.output_path, state_dict=deloreanized_sd)
        tokenizer.save_pretrained(args.output_path)
    except Exception as e:
        print(f"Failed to merge the adapter, error: {e}.")
    finally:
        if qa_lora and temp_dir:
           temp_dir.cleanup()
