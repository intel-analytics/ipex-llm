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

from transformers import LlamaTokenizer
from ipex_llm.transformers import AutoModelForCausalLM
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save model with bigdl-llm low-bit optimization')
    parser.add_argument('--base_model', type=str, default="meta-llama/Llama-2-70b-hf",
                        help='The huggingface repo id for the Llama2-70B model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--output_path', type=str, default="./llama-2-70b-hf-nf4",
                        help='The path to the saved model.')

    args = parser.parse_args()
    base_model = args.base_model
    output_path = args.output_path

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_low_bit="nf4",
        # load_in_4bit=True,
        optimize_model=False,
        torch_dtype=torch.bfloat16,
        # device_map=device_map,
        modules_to_not_convert=["lm_head"],
    )

    model.save_low_bit(output_path)
    print(f'Model with bigdl-llm low-bit optimization is saved to {output_path}.')
