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
import time
import argparse
from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import warnings

# you could tune the prompt based on your own model,
# here the prompt tuning refers to https://huggingface.co/georgesung/llama2_7b_chat_uncensored#prompt-style
PROMPT_FORMAT = """### HUMAN:
{prompt}

### RESPONSE:
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for LLM model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help='The huggingface repo id'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="What is AI?",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    warnings.warn("iq2 quantization may need several minutes, please wait a moment, "
                  "or have a cup of coffee now : )")

    # Load model in 2 bit,
    # which convert the relevant layers in the model into gguf_iq2_xxs format.
    # GGUF-IQ2 quantization needs imatrix file to assist in quantization
    # and improve generation quality, and different model may need different
    # imtraix file, you can find and download imatrix file from 
    # https://huggingface.co/datasets/ikawrakow/imatrix-from-wiki-train/tree/main.
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 load_in_low_bit='gguf_iq2_xxs',
                                                 torch_dtype=torch.float16,
                                                 trust_remote_code=True,
                                                 imatrix='llama-v2-7b.imatrix').to("xpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Generate predicted tokens
    with torch.inference_mode():
        prompt = PROMPT_FORMAT.format(prompt=args.prompt)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to("xpu")
        # ipex_llm model needs a warmup, then inference time can be accurate
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict)
        st = time.time()
        # if your selected model is capable of utilizing previous key/value attentions
        # to enhance decoding speed, but has `"use_cache": false` in its model config,
        # it is important to set `use_cache=True` explicitly in the `generate` function
        # to obtain optimal performance with IPEX-LLM Low Bit optimizations
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict,
                                repetition_penalty=1.1)
        end = time.time()
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Prompt', '-'*20)
        print(prompt)
        print('-'*20, 'Output', '-'*20)
        print(output_str)
