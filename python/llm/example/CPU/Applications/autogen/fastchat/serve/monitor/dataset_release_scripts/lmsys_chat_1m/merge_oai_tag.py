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
import argparse
import json
import time

from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--sample", type=int)
    args = parser.parse_args()

    tag_file = "clean_conv_20230809_1.5M_oai_filter_v2.json"
    # tag_file = "clean_conv_20230809_1.5M_oai_filter_v2_100k.json"
    in_file = args.in_file
    tic = time.time()

    # Load tags
    print("Load tags...")
    tag_data = json.load(open(tag_file))
    tag_dict = {}
    for c in tqdm(tag_data):
        tag_dict[c["conversation_id"]] = [x["oai_filter"] for x in c["conversation"]]
    print(f"elapsed: {time.time() - tic:.2f} s")

    # Append to input_file
    print("Load inputs...")
    input_data = json.load(open(in_file))
    for c in tqdm(input_data):
        cid = c["conversation_id"]
        if cid in tag_dict:
            c["openai_moderation"] = tag_dict[cid]
        else:
            print(f"missing tag for conv {cid}")
            exit()
    print(f"elapsed: {time.time() - tic:.2f} s")

    # Write output
    print("Write outputs...")
    out_file = in_file.replace(".json", ".with_tag.json")
    print(f"Output to {out_file}")
    with open(out_file, "w") as fout:
        json.dump(input_data, fout, indent=2, ensure_ascii=False)
    print(f"elapsed: {time.time() - tic:.2f} s")
