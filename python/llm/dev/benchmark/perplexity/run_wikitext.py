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
# https://huggingface.co/docs/transformers/en/perplexity
#

import argparse
import torch
from tqdm import tqdm
from datasets import concatenate_datasets, load_dataset
from ppl import BigDLPPL
from ipex_llm.ggml.quantize import ggml_tensor_qtype
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--wikitext_mode", required=False, type=str, default='disable')
parser.add_argument("--model_path", required=True, type=str)
parser.add_argument("--datasets", required=False, type=str, default=None, nargs='*')
parser.add_argument("--dataset_path", required=False, type=str, default=None)
parser.add_argument("--stride", type=int, default=0)
parser.add_argument("--device", type=str, default="xpu")
parser.add_argument("--precision", type=str, default="sym_int4")
parser.add_argument("--use-cache", action="store_true")
parser.add_argument("--language", required=False, type=str, default="en", choices=['en', 'zh', 'all'])
parser.add_argument("--output_path", default=None)
parser.add_argument("--stride", type=int, default=512)
args = parser.parse_args()

if args.wikitext_mode != 'disable':
    if args.precision == "fp16":  # ipex fp16
        from transformers import AutoModelForCausalLM
        if "xpu" in args.device:
            import intel_extension_for_pytorch as ipex
        model = AutoModelForCausalLM.from_pretrained(args.model_path, use_cache=args.use_cache, trust_remote_code=True)
        model = model.half()
    else:  # ipex-llm
        from ipex_llm.transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.model_path, load_in_low_bit=args.precision,
                                                    use_cache=args.use_cache, trust_remote_code=True)
        model = model.half()
    model = model.to(args.device)
    model = model.eval()

    if args.wikitext_mode == 'new':
        with open('/home/wangyishuo/liuzicheng/source/wikitext-2-raw-v1/wikitext-2-raw/wiki.test.raw', "rb") as f:
            data = f.read()
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        encodings = tokenizer(data.decode("utf-8").strip("\n"), return_tensors="pt")
    
    if args.wikitext_mode == 'old':
        def parse_kwargs(kwstr):
            kvpair = [item.split('=') for item in kwstr.split(',') if item != ""]
            return {k:v for k, v in kvpair}
        
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        test = load_dataset(**parse_kwargs('path=wikitext,name=wikitext-2-raw-v1'), split="test")["text"]
        encodings = tokenizer("\n\n".join(test), return_tensors="pt")

    max_length = model.config.max_position_embeddings
    stride = args.chunk_size if args.stride <= 0 else args.stride
    seq_len = encodings.input_ids.size(1)
    num_chunks = seq_len // stride

    nlls = []
    prev_end_loc = 0
    for i in tqdm(range(num_chunks)):
        begin_loc = i * stride
        if args.stride > 0:
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        else:
            end_loc = begin_loc + stride
            trg_len = -stride//2
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(args.device)
        if args.stride == 0: input_ids[:, 0] = tokenizer.bos_token_id
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        if "xpu" in args.device:
            torch.xpu.empty_cache()

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    print("Final ppl estimate: {}".format(ppl.item()))

else:

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
        ppl_evaluator = BigDLPPL(model_path=args.model_path, device=args.device, **model_kwargs)
        ppl = ppl_evaluator.perplexity_hf(encoded_texts)
        summary[precision] = ppl
        results['results'] = ppl
        results['config'] = {"model": model_name, "precision": precision, "device": args.device, "seq_len": args.seq_len, "language": args.language}
        dumped = json.dumps(results, indent=2)
        print(dumped)

        if output_path:
            with open(f"{log_dir}/result.json", "w") as f:
                f.write(dumped)
    
    print(summary)