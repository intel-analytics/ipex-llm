
import torch
import time
import argparse
import numpy as np
from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, GenerationConfig
import intel_extension_for_pytorch as ipex
# you could tune the prompt based on your own model,
# here the prompt tuning refers to https://huggingface.co/internlm/internlm-chat-7b-8k/blob/main/modeling_internlm.py#L768
PROMPT_FORMAT = "<|User|>:{prompt}\n<|Bot|>:"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for InternLM model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="/mnt/disk1/models/deepseek-moe-16b-chat",
                        help='The huggingface repo id for the InternLM model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="What is AI",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    # When running LLMs on Intel iGPUs for Windows users, we recommend setting `cpu_embedding=True` in the from_pretrained function.
    # This will allow the memory-intensive embedding layer to utilize the CPU instead of iGPU.
    # from bigdl.llm.transformers import AutoModelForCausalLM
    # model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, load_in_4bit=True).eval()
    # model.generation_config = GenerationConfig.from_pretrained(model_path)
    # model.generation_config.pad_token_id = model.generation_config.eos_token_id
    
    # optimize model
    from transformers import AutoModelForCausalLM
    from bigdl.llm import optimize_model
    model = AutoModelForCausalLM.from_pretrained(model_path).eval()
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model = optimize_model(model)
    
    model = model.to('xpu')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    
    # Generate predicted tokens
    with torch.inference_mode():
        prompt = PROMPT_FORMAT.format(prompt=args.prompt)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu')
        # ipex model needs a warmup, then inference time can be accurate
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict)

        # start inference
        cost = []
        for _ in range(3):
            st = time.time()
            # if your selected model is capable of utilizing previous key/value attentions
            # to enhance decoding speed, but has `"use_cache": false` in its model config,
            # it is important to set `use_cache=True` explicitly in the `generate` function
            # to obtain optimal performance with BigDL-LLM INT4 optimizations
            output = model.generate(input_ids,
                                    max_new_tokens=args.n_predict)
            torch.xpu.synchronize()
            end = time.time()
            cost.append(end-st)
            output = output.cpu()
            output_str = tokenizer.decode(output[0], skip_special_tokens=True)
            output_str = output_str.split("<eoa>")[0]
            print('-'*20, 'Prompt', '-'*20)
            print(prompt)
            print('-'*20, 'Output', '-'*20)
            print(output_str)
        ave_time = np.average(cost)
        print(ave_time)

