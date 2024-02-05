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
from tqdm import tqdm
import torch
from datasets import concatenate_datasets, load_from_disk
from transformers import AutoTokenizer

from ppl import BigDLPPL
from bigdl.llm.ggml.quantize import ggml_tensor_qtype
from bigdl.llm.transformers import AutoModelForCausalLM, AutoModel

import os

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--precisions", required=False, type=str, default=None, nargs='+')
    parser.add_argument("--device", type=str, default="xpu")
    return parser.parse_args()
    

@torch.no_grad()
def main():
    args = get_arguments()
    for arg_name, arg_var in args.__dict__.items():
        print(f"{arg_name:<16} : {arg_var}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.model_max_length = args.seq_len

    dataset_paths = [os.path.join(args.dataset, d) for d in os.listdir(args.dataset)]
    data_subset_all = []
    for dataset_path in dataset_paths:
        data_ = load_from_disk(dataset_path)       
        data_subset_all.append(data_)
    data = concatenate_datasets(data_subset_all)

    encoded_texts = []
    pbar = tqdm(data)
    for i, data_i in enumerate(pbar):
        encoded_text = tokenizer.encode(data_i['context'], return_tensors='pt', truncation=True)
        pbar.set_description(f"seq_len: {len(encoded_text[0])}, n_data: {len(encoded_texts)}")
        if len(encoded_text[0]) < args.seq_len:
            continue
        encoded_texts.append(encoded_text)
    
    summary = {}
    for precision in args.precisions:
        model_kwargs = {}
        if precision in ggml_tensor_qtype.keys():
            model_kwargs['load_in_low_bit'] = precision
        else:
            model_kwargs['torch_dtype'] = getattr(torch, precision)
        print(model_kwargs)
        
        ppl_evaluator = BigDLPPL(model_path=args.model_path, device=args.device, **model_kwargs)
        ppl = ppl_evaluator.perplexity_hf(encoded_texts)
        summary[precision] = ppl
    print(summary)

if __name__ == "__main__":
    main()