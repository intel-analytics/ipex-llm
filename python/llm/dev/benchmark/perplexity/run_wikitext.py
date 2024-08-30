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
from datasets import load_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True, type=str)
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--data_path", type=str, default=None)
parser.add_argument("--chunk_size", type=int, default=512)
parser.add_argument("--stride", type=int, default=0)
parser.add_argument("--device", type=str, default="xpu")
parser.add_argument("--precision", type=str, default="sym_int4")
parser.add_argument("--use-cache", action="store_true")
parser.add_argument("--max_length", type=int, default=None)
parser.add_argument("--mixed_precision", action="store_true") 
args = parser.parse_args()

if args.precision == "fp16":  # ipex fp16
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 use_cache=args.use_cache,
                                                 trust_remote_code=True)
    model = model.half()
elif 'gptq' in args.model_path.lower():  # ipex-llm gptq
    from ipex_llm.transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 load_in_4bit=True,
                                                 torch_dtype=torch.float,
                                                 use_cache=args.use_cache,
                                                 trust_remote_code=True)
else:  # ipex-llm
    from ipex_llm.transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 load_in_low_bit=args.precision,
                                                 use_cache=args.use_cache,
                                                 trust_remote_code=True,
                                                 mixed_precision=args.mixed_precision)   
    model = model.half()
model = model.to(args.device)
model = model.eval()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

if args.dataset:
    def parse_kwargs(kwstr):
        kvpair = [item.split('=') for item in kwstr.split(',') if item != ""]
        return {k:v for k, v in kvpair}
    test = load_dataset(**parse_kwargs(args.dataset), split="test")["text"]
    encodings = tokenizer("\n\n".join(test), return_tensors="pt")
elif args.data_path:
    with open(args.data_path, "rb") as f:
        data = f.read()
    encodings = tokenizer(data.decode("utf-8").strip("\n"), return_tensors="pt")
else:
    from ipex_llm.utils.common import invalidInputError
    raise invalidInputError(False, "Must specify either dataset or datapath.")

if not args.max_length:
    try:
        max_length = model.config.max_position_embeddings
    except:
        max_length = model.config.seq_length  # max_length in config of chatglm is 'seq_length'
else:
    max_length = args.max_length
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
