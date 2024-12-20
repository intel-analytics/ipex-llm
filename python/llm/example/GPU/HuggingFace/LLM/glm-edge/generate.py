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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for GLM-Edge model')
    parser.add_argument('--repo-id-or-model-path', type=str,
                        help='The  Hugging Face or ModelScope repo id for the GLM-Edge model to be downloaded'
                             ', or the path to the checkpoint folder')
    parser.add_argument('--prompt', type=str, default="AI是什么？",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')
    parser.add_argument('--modelscope', action="store_true", default=False, 
                        help="Use models from modelscope")


    args = parser.parse_args()

    if args.modelscope:
        from modelscope import AutoTokenizer
        model_hub = 'modelscope'
    else:
        from transformers import AutoTokenizer
        model_hub = 'huggingface'
    
    model_path = args.repo_id_or_model_path if args.repo_id_or_model_path else \
        ("ZhipuAI/glm-edge-4b-chat" if args.modelscope else "THUDM/glm-edge-4b-chat")

    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    # When running LLMs on Intel iGPUs for Windows users, we recommend setting `cpu_embedding=True` in the from_pretrained function.
    # This will allow the memory-intensive embedding layer to utilize the CPU instead of iGPU.
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 load_in_4bit=True,
                                                 optimize_model=True,
                                                 trust_remote_code=True,
                                                 use_cache=True,
                                                 model_hub=model_hub)
    model = model.half().to("xpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    
    # Generate predicted tokens
    with torch.inference_mode():
        # The following code for generation is adapted from https://huggingface.co/THUDM/glm-edge-1.5b-chat#inference
        message = [{"role": "user", "content": args.prompt}]

        inputs = tokenizer.apply_chat_template(
            message,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        ).to("xpu")
        
        generate_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": args.n_predict,
            "do_sample": False,
        }
    
        # ipex_llm model needs a warmup, then inference time can be accurate
        output = model.generate(**generate_kwargs)

        st = time.time()
        output = model.generate(**generate_kwargs)
        torch.xpu.synchronize()
        end = time.time()

        output_str = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Prompt', '-'*20)
        print(args.prompt)
        print('-'*20, 'Output', '-'*20)
        print(output_str)
