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


def convert_and_load(repo_id_or_model_path, model_family, n_threads):

    from bigdl.llm.ggml.transformers import AutoModelForCausalLM

    # here you may input the HuggingFace repo id directly as the value of `pretrained_model_name_or_path`.
    # This will allow the pre-trained model to be downloaded directly from the HuggingFace repository.
    # The downloaded model will then be converted to binary format with int4 dtype weights,
    # and saved into the cache_dir folder.
    #
    # if you already have the pre-trained model downloaded, you can provide the path to
    # the downloaded folder as the value of `pretrained_model_name_or_path``
    llm = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=repo_id_or_model_path,
        model_family=model_family,
        dtype='int4',
        cache_dir='./',
        n_threads=n_threads)

    # if you want to explicitly convert the pre-trained model, you can use the `llm_convert` API 
    # to convert the downloaded Huggungface checkpoint first,
    # and then load the binary checkpoint directly.
    #
    # from bigdl.llm import llm_convert
    #
    # model_path = repo_id_or_model_path
    # output_ckpt_path = llm_convert(
    #     model=model_path,
    #     outfile='./',
    #     outtype='int4',
    #     model_family=model_family)
    #
    # llm = AutoModelForCausalLM.from_pretrained(
    #     pretrained_model_name_or_path=output_ckpt_path,
    #     model_family=model_family,
    #     n_threads=n_threads)

    return llm

def inference(llm, repo_id_or_model_path, model_family, prompt):

    if model_family in ['llama', 'gptneox']:
        # Option 1: Use HuggingFace transformers tokenizer
        print('-'*20, ' HuggingFace transformers tokenizer ', '-'*20)
        
        print('Please note that the loading of HuggingFace transformers tokenizer may take some time.\n')
        # here is only a workaround for default example model 'decapoda-research/llama-7b-hf' in LLaMA family,
        # due to its out-of-date 'tokenizer_class' defined in its tokenizer_config.json.
        #
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

        # Option 2: Use bigdl-llm based tokenizer
        print('-'*20, ' bigdl-llm based tokenizer ', '-'*20)
        st = time.time()

        # please note that the prompt here can either be a string or a list of string
        tokens_id = llm.tokenize(prompt)
        output_tokens_id = llm.generate(tokens_id, max_new_tokens=32)
        output = llm.batch_decode(output_tokens_id)

        print(f'Inference time: {time.time()-st} s')
        print(f'Output:\n{output}')

    if model_family in ['llama', 'gptneox', 'bloom', 'starcoder']:
        # Option 3: fast forward
        # note that currently Bloom/Starcoder family model only supports the fast forward inference method
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
                        help="The model family of the large language model (supported option: 'llama', "
                             "'gptneox', 'bloom', 'starcoder')")
    parser.add_argument('--repo-id-or-model-path', type=str,
                        help='The huggingface repo id for the larga language model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default='Q: What is CPU? A:',
                        help='Prompt to infer')
    args = parser.parse_args()

    repo_id_or_model_path = args.repo_id_or_model_path
    if args.repo_id_or_model_path is None:
        if args.model_family == 'llama':
            repo_id_or_model_path = 'decapoda-research/llama-7b-hf'
        elif args.model_family == 'gptneox':
            repo_id_or_model_path = 'togethercomputer/RedPajama-INCITE-7B-Chat'
        elif args.model_family == 'bloom':
            repo_id_or_model_path = 'bigscience/bloomz-7b1'
        elif args.model_family == 'starcoder':
            repo_id_or_model_path = 'bigcode/gpt_bigcode-santacoder'

    # Step 1: convert and load int4 model
    llm = convert_and_load(repo_id_or_model_path=repo_id_or_model_path,
                           model_family=args.model_family,
                           n_threads=args.thread_num)

    # Step 2: conduct inference
    inference(llm=llm,
              repo_id_or_model_path=repo_id_or_model_path,
              model_family=args.model_family,
              prompt=args.prompt)


if __name__ == '__main__':
    main()
