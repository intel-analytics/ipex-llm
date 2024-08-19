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
# This file is adapted from
# https://github.com/insuhan/hyper-attn/blob/main/benchmark_patch_llm.py
#

import argparse
from tqdm import tqdm
import torch
from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer

from ppl import BigDLPPL
from ipex_llm.ggml.quantize import ggml_tensor_qtype

import os
import json

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--datasets", required=False, type=str, default=None, nargs='*')
    parser.add_argument("--dataset_path", required=False, type=str, default=None)
    parser.add_argument("--language", required=False, type=str, default="en", choices=['en', 'zh', 'all'])
    parser.add_argument("--precisions", required=False, type=str, default=None, nargs='+')
    parser.add_argument("--mixed_precision", action="store_true") 
    parser.add_argument("--device", type=str, default="xpu")
    parser.add_argument("--output_path", default=None)
    return parser.parse_args()
    

@torch.no_grad()
def main():
    args = get_arguments()
    for arg_name, arg_var in args.__dict__.items():
        print(f"{arg_name:<16} : {arg_var}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.model_max_length = args.seq_len

    en_datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "gov_report", 
                       "qmsum", "multi_news",  "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en"]
    zh_datasets = ["multifieldqa_zh", "dureader", "vcsum", "lsht", "passage_retrieval_zh"]

    if args.datasets is None:
        if args.language == 'en':
            datasets = en_datasets
        elif args.language == 'zh':
            datasets = zh_datasets
        else:
            datasets = en_datasets + zh_datasets
    else:
        datasets = args.datasets

    dataset_all = []
    for dataset_name in datasets:
        data_ = load_dataset(os.path.join(args.dataset_path, dataset_name), split='test') if args.dataset_path \
                else load_dataset('THUDM/LongBench', f'{dataset_name}', split='test')
        dataset_all.append(data_)
    data = concatenate_datasets(dataset_all)

    encoded_texts = []
    pbar = tqdm(data)
    for i, data_i in enumerate(pbar):
        encoded_text = tokenizer.encode(data_i['context'], return_tensors='pt', truncation=True)
        pbar.set_description(f"seq_len: {len(encoded_text[0])}, n_data: {len(encoded_texts)}")
        if len(encoded_text[0]) < args.seq_len:
            continue
        encoded_texts.append(encoded_text)
    
    summary = {}
    output_path = args.output_path if args.output_path else "results"
    model_name = os.path.basename(os.path.realpath(args.model_path))
    for precision in args.precisions:
        model_kwargs = {}
        if precision in ggml_tensor_qtype.keys():
            model_kwargs['load_in_low_bit'] = precision
        else:
            model_kwargs['torch_dtype'] = getattr(torch, precision)
        print(model_kwargs)
        
        log_dir = f"{output_path}/{model_name}/{args.device}/{precision}/{args.language}"
        os.makedirs(log_dir, exist_ok=True)
        results = {}
        ppl_evaluator = BigDLPPL(model_path=args.model_path, device=args.device, mixed_precision=args.mixed_precision, **model_kwargs)
        ppl = ppl_evaluator.perplexity_hf(encoded_texts)
        summary[precision] = ppl
        results['results'] = ppl
        results['config'] = {"model": model_name, "precision": precision, "mixed_precision": args.mixed_precision, "device": args.device, "seq_len": args.seq_len, "language": args.language }
        dumped = json.dumps(results, indent=2)
        print(dumped)

        if output_path:
            with open(f"{log_dir}/result.json", "w") as f:
                f.write(dumped)
    
    print(summary)

if __name__ == "__main__":
    main()