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
from bigdl.llm.ggml.quantize import ggml_tensor_qtype
from ppl import BigDLPPL
from datasets import load_dataset
import argparse


def parse_kwargs(kwstr):
    kvpair = [item.split('=') for item in kwstr.split(',') if item != ""]
    return {k:v for k, v in kvpair}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--precisions", required=False, type=str, default=None, nargs='+')
    parser.add_argument("--model_kwargs", required=False, type=str, default="")
    parser.add_argument("--torch_dtype", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dataset", type=str, default='path=wikitext,name=wikitext-2-raw-v1')

    return parser.parse_args()


def main():
    args = parse_args()
    text = load_dataset(**parse_kwargs(args.dataset), split="test")["text"]
    additional_model_kwargs = parse_kwargs(args.model_kwargs)
    summary = {}
    for precision in args.precisions:
        model_kwargs = additional_model_kwargs.copy()
        if precision in ggml_tensor_qtype.keys():
            model_kwargs['load_in_low_bit'] = precision
        else:
            model_kwargs['torch_dtype'] = getattr(torch, precision)
        print(model_kwargs)
        ppl_evaluator = BigDLPPL(model_path=args.model_path, device=args.device, **model_kwargs)
        ppl = ppl_evaluator.perplexity_hf(text)
        summary[precision] = ppl
    print(summary)

main()