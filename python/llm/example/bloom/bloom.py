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


def convert_and_load(repo_id_or_model_path, n_threads):

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
        model_family='bloom',
        dtype='int4',
        cache_dir='./',
        n_threads=n_threads)

    # if you want to explicitly convert the pre-trained model, you can use the `convert_model` API 
    # to convert the downloaded Huggungface checkpoint first,
    # and then load the binary checkpoint directly.
    #
    # from bigdl.llm.ggml import convert_model
    #
    # model_path = repo_id_or_model_path
    # output_ckpt_path = convert_model(
    #     input_path=model_path,
    #     output_path='./',
    #     dtype='int4',
    #     model_family='bloom')
    #
    # llm = AutoModelForCausalLM.from_pretrained(
    #     pretrained_model_name_or_path=output_ckpt_path,
    #     model_family='bloom',
    #     n_threads=n_threads)

    return llm

def inference(llm, prompt, repo_id_or_model_path):

    st = time.time()

    output = llm(prompt,
                 max_tokens=32)

    print(f'Inference time (fast forward): {time.time()-st} s')
    print(f'Output:\n{output}')


def main():
    parser = argparse.ArgumentParser(description='BLOOM pipeline example')
    parser.add_argument('--thread-num', type=int, default=2, required=True,
                        help='Number of threads to use for inference')
    parser.add_argument('--repo-id-or-model-path', type=str, default="bigscience/bloomz-7b1",
                        help='The huggingface repo id for BLOOM family model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default='Q: tell me something about Intel. A:',
                        help='Prompt to infer')
    args = parser.parse_args()

    # Step 1: convert and load int4 model
    llm = convert_and_load(repo_id_or_model_path=args.repo_id_or_model_path, n_threads=args.thread_num)

    # Step 2: conduct inference
    inference(llm=llm, prompt=args.prompt, repo_id_or_model_path=args.repo_id_or_model_path)


if __name__ == '__main__':
    main()
