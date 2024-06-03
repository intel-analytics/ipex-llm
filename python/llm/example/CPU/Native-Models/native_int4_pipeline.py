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


import time
import argparse
from ipex_llm.transformers import *


def convert(repo_id_or_model_path, model_family, tmp_path):
    from ipex_llm import llm_convert
    original_llm_path = repo_id_or_model_path
    bigdl_llm_path = llm_convert(
        model=original_llm_path,
        outfile='./',
        outtype='int4',
        tmp_path=tmp_path,
        model_family=model_family)

    return bigdl_llm_path

def load(model_path, model_family, n_threads):
    model_family_to_class = {
        "llama": LlamaForCausalLM,
        "gptneox": GptneoxForCausalLM,
        "bloom": BloomForCausalLM,
        "starcoder": StarcoderForCausalLM
    }

    if model_family in model_family_to_class:
        llm_causal = model_family_to_class[model_family]
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    llm = llm_causal.from_pretrained(
        pretrained_model_name_or_path=model_path,
        native=True,
        dtype="int4",
        n_threads=n_threads)

    return llm

def inference(llm, repo_id_or_model_path, model_family, prompt):

    if model_family in ['llama', 'gptneox', 'bloom', 'starcoder']:
        # ------ Option 1: Use IPEX-LLM based tokenizer
        print('-'*20, ' IPEX-LLM based tokenizer ', '-'*20)
        st = time.time()

        # please note that the prompt here can either be a string or a list of string
        tokens_id = llm.tokenize(prompt)
        output_tokens_id = llm.generate(tokens_id, max_new_tokens=32)
        output = llm.batch_decode(output_tokens_id)
        
        print(f'Inference time: {time.time()-st} s')
        print(f'Output:\n{output}')
        
        # ------- Option 2: Use HuggingFace transformers tokenizer
        print('-'*20, ' HuggingFace transformers tokenizer ', '-'*20)
        
        print('Please note that the loading of HuggingFace transformers tokenizer may take some time.\n')
        # here is only a workaround for default example model 'decapoda-research/llama-7b-hf' in LLaMA family,
        # due to its out-of-date 'tokenizer_class' defined in its tokenizer_config.json.
        
        # for most cases, you could use `AutoTokenizer`.
        if model_family == 'llama':
            from transformers import LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(repo_id_or_model_path)
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(repo_id_or_model_path)

        st = time.time()

        # please note that the prompt here can either be a string or a list of string
        tokens_id = tokenizer(prompt).input_ids
        output_tokens_id = llm.generate(tokens_id, max_new_tokens=32)
        output = tokenizer.batch_decode(output_tokens_id)

        print(f'Inference time: {time.time()-st} s')
        print(f'Output:\n{output}')

        # Option 3: fast forward
        print('-'*20, ' fast forward ', '-'*20)
        st = time.time()

        output = llm(prompt, # please note that the prompt here can ONLY be a string
                     max_tokens=32)

        print(f'Inference time (fast forward): {time.time()-st} s')
        print(f'Output:\n{output}')


def main():
    parser = argparse.ArgumentParser(description='INT4 pipeline example')
    parser.add_argument('--thread-num', type=int, default=2, required=True,
                        help='Number of threads to use for inference')
    parser.add_argument('--model-family', type=str, default='llama', required=True,
                        choices=["llama", "llama2", "bloom", "gptneox", "starcoder"],
                        help="The model family of the large language model (supported option: 'llama', 'llama2', "
                             "'gptneox', 'bloom', 'starcoder')")
    parser.add_argument('--repo-id-or-model-path', type=str, required=True,
                        help='The path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default='Once upon a time, there existed a little girl who liked to have adventures. ',
                        help='Prompt to infer')
    parser.add_argument('--tmp-path', type=str, default='/tmp',
                        help='path to store intermediate model during the conversion process')
    args = parser.parse_args()

    repo_id_or_model_path = args.repo_id_or_model_path

    # Currently, we can directly use llama related implementation to run llama2 models
    if args.model_family == 'llama2':
        args.model_family = 'llama'

    # Step 1: convert original model to IPEX-LLM model
    ipex_llm_path = convert(repo_id_or_model_path=repo_id_or_model_path,
                             model_family=args.model_family,
                             tmp_path=args.tmp_path)
    
    # Step 2: load int4 model
    llm = load(model_path=ipex_llm_path,
               model_family=args.model_family,
               n_threads=args.thread_num)

    # Step 3: inference
    inference(llm=llm,
              repo_id_or_model_path=repo_id_or_model_path,
              model_family=args.model_family,
              prompt=args.prompt)


if __name__ == '__main__':
    main()
