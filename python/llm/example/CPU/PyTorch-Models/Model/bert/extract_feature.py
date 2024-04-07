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

from transformers import BertTokenizer, BertModel
from ipex_llm import optimize_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract the feature of given text using BERT model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="bert-large-uncased",
                        help='The huggingface repo id for the BERT (e.g. `bert-large-uncased`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--text', type=str, default="This is an example text for feature extraction.",
                        help='Text to extract features')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    # Load model
    model = BertModel.from_pretrained(model_path,
                                      torch_dtype="auto",
                                      low_cpu_mem_usage=True)
    
    # With only one line to enable IPEX-LLM optimization on model
    model = optimize_model(model)

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    # Extract the feature of given text
    text = args.text
    encoded_input = tokenizer(text, return_tensors='pt')
    st = time.time()
    output = model(**encoded_input)
    end = time.time()
    print(f'Time cost: {end-st} s')
    print('-'*20, 'Output', '-'*20)
    print(output)
