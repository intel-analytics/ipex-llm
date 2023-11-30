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
Usage:
python3 -m fastchat.data.inspect_data --in sharegpt_20230322_clean_lang_split.json
"""
import argparse
import json
import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--begin", type=int)
    parser.add_argument("--random-n", type=int)
    args = parser.parse_args()

    content = json.load(open(args.in_file, "r"))

    if args.random_n:
        indices = [random.randint(0, len(content) - 1) for _ in range(args.random_n)]
    elif args.begin:
        indices = range(args.begin, len(content))
    else:
        indices = range(0, len(content))

    for idx in indices:
        sample = content[idx]
        print("=" * 40)
        print(f"no: {idx}, id: {sample['id']}")
        for conv in sample["conversations"]:
            print(conv["from"] + ": ")
            print(conv["value"])
            input()
