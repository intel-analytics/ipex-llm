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
Take the intersection of two conversation files.

Usage: python3 -m fastchat.data.merge --input input.json --conv-id conv_id_file.json --out intersect.json
"""

import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--conv-id", type=str, required=True)
    parser.add_argument("--out-file", type=str, default="intersect.json")
    args = parser.parse_args()

    conv_id_objs = json.load(open(args.conv_id, "r"))
    conv_ids = set(x["conversation_id"] for x in conv_id_objs)

    objs = json.load(open(args.input, "r"))
    after_objs = [x for x in objs if x["conversation_id"] in conv_ids]

    print(f"#in: {len(objs)}, #out: {len(after_objs)}")
    json.dump(after_objs, open(args.out_file, "w"), indent=2, ensure_ascii=False)
