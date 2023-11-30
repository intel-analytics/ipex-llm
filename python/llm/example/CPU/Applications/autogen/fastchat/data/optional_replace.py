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
# Copyright 2023 The FastChat team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""
Do optional replace of bos/eos/pad/unk.

Usage:
python3 -m fastchat.data.optional_replace --in input.json --out output.json --model-name-or-path <your_token_path>

Requirement:
pip3 install transformers tqdm
"""
import argparse
import json
import traceback

import transformers
from tqdm import tqdm


def replace_special_tokens(
    tokenizer: transformers.PreTrainedTokenizer, text: str
) -> str:
    if not text:
        return text

    def _insert_vline(token: str) -> str:
        if len(token) < 2:
            return " "
        elif len(token) == 2:
            return f"{token[0]}|{token[1]}"
        else:
            return f"{token[:1]}|{token[1:-1]}|{token[-1:]}"

    if tokenizer.bos_token:
        text = text.replace(tokenizer.bos_token, _insert_vline(tokenizer.bos_token))
    if tokenizer.eos_token:
        text = text.replace(tokenizer.eos_token, _insert_vline(tokenizer.eos_token))
    if tokenizer.pad_token:
        text = text.replace(tokenizer.pad_token, _insert_vline(tokenizer.pad_token))
    if tokenizer.unk_token:
        text = text.replace(tokenizer.unk_token, _insert_vline(tokenizer.unk_token))
    return text


def replace(conv, tokenizer):
    # Replace bos/eos/pad/unk tokens
    if tokenizer:
        try:
            for sentence in conv["conversations"]:
                sentence["value"] = replace_special_tokens(tokenizer, sentence["value"])
        except Exception as e:
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str)
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        help="The directory or address where the model token is stored.",
    )
    args = parser.parse_args()

    in_file = args.in_file
    out_file = args.out_file
    tokenizer = None
    if args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            use_fast=False,
        )

    if out_file is None:
        out_file = f"{in_file}_replace.json"

    content = json.load(open(in_file, "r"))

    for conv in tqdm(content):
        replace(conv, tokenizer)

    json.dump(content, open(out_file, "w"), indent=2, ensure_ascii=False)
