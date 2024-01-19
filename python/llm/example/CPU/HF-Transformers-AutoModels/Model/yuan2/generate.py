import torch, transformers
import sys, os, time
import argparse
from transformers import LlamaTokenizer
from bigdl.llm.transformers import AutoModelForCausalLM

# Refer to https://huggingface.co/IEITYuan/Yuan2-2B-hf#Usage
YUAN2_PROMPT_FORMAT = """
{prompt}
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text using Yuan2-2B model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="IEITYuan/Yuan2-2B-hf",
                        help='The huggingface repo id for the Yuan2 to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="请问目前最先进的机器学习算法有哪些？",
                        help='Prompt for the model')
    parser.add_argument('--n-predict', type=int, default=100,
                        help='Number of tokens to generate')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    # Load tokenizer
    print("Creating tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(model_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
    tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>',
                          '<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    print("Creating model...")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", trust_remote_code=True, load_in_4bit=True).eval()

    prompt = YUAN2_PROMPT_FORMAT.format(prompt=args.prompt)
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]

    # Measure the inference time
    start_time = time.time()
    # if your selected model is capable of utilizing previous key/value attentions
    # to enhance decoding speed, but has `"use_cache": false` in its model config,
    # it is important to set `use_cache=True` explicitly in the `generate` function
    # to obtain optimal performance with BigDL-LLM INT4 optimizations
    outputs = model.generate(inputs, do_sample=True, top_k=5, max_length=args.n_predict)
    end_time = time.time()

    output_str = tokenizer.decode(outputs[0])
    print(f'Inference time: {end_time - start_time} seconds')
    print('-'*20, 'Output', '-'*20)
    print(output_str)