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
Filter conversations with wrong formats.

Usage:
python3 -m fastchat.data.filter_wrong_format --in input.json --out output.json

"""
import argparse
import json
import re

from tqdm import tqdm

wrong_indices_pattern = re.compile("\n1\. [^2]*\n1\. ")


def should_skip(conv):
    # Filter wrong list indices like https://sharegpt.com/c/1pREAGO
    for sentence in conv["conversations"]:
        val = sentence["value"]
        sub = re.search(wrong_indices_pattern, val)
        if sub is not None:
            return True

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, required=True)
    args = parser.parse_args()

    content = json.load(open(args.in_file, "r"))

    new_content = []
    for conv in tqdm(content):
        if should_skip(conv):
            print(f"{conv['id']} contains a wrong format.")
        else:
            new_content.append(conv)

    print(f"#in: {len(content)}, #out: {len(new_content)}")
    json.dump(new_content, open(args.out_file, "w"), indent=2, ensure_ascii=False)
