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

import argparse
from ipex_llm.transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer, TextGenerationPipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer save_load example')
    parser.add_argument('--repo-id-or-model-path', type=str, default="decapoda-research/llama-7b-hf",
                        help='The huggingface repo id for the large language model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--low-bit', type=str, default="sym_int4",
                        choices=['sym_int4', 'asym_int4', 'sym_int5', 'asym_int5', 'sym_int8'],
                        help='The quantization type the model will convert to.')
    parser.add_argument('--save-path', type=str, default=None,
                        help='The path to save the low-bit model.')
    parser.add_argument('--load-path', type=str, default=None,
                        help='The path to load the low-bit model.')
    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    low_bit = args.low_bit
    load_path = args.load_path
    if load_path:
        model = AutoModelForCausalLM.load_low_bit(load_path)
        tokenizer = LlamaTokenizer.from_pretrained(load_path)
    else:
        # load_in_low_bit in ipex_llm.transformers will convert
        # the relevant layers in the model into corresponding int X format
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit=low_bit, trust_remote_code=True)
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)

    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, max_new_tokens=32)
    input_str = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"
    output = pipeline(input_str)[0]["generated_text"]
    print(f"Prompt: {input_str}")
    print(f"Output: {output}")

    save_path = args.save_path
    if save_path:
        model.save_low_bit(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model and tokenizer are saved to {save_path}")
