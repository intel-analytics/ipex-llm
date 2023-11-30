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
"""Count the unique users in a battle log file."""

import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--tag-file", type=str)
    args = parser.parse_args()

    # build index
    objs = json.load(open(args.tag_file))
    new_field_dict = {}
    for obj in objs:
        new_field_dict[obj["question_id"]] = obj["toxic_chat"]

    objs = json.load(open(args.input))
    for obj in objs:
        obj["toxic_chat_tag"] = new_field_dict[obj["question_id"]]

    output = args.input.replace(".json", "_added.json")
    with open(output, "w") as fout:
        json.dump(objs, fout, indent=2, ensure_ascii=False)
