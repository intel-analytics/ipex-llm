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

import json
from argparse import ArgumentParser
import sys
import os

parser = ArgumentParser()
parser.add_argument('--input_meta', type=str, required=True,
                    help="item metadata file")

args = parser.parse_args()
# process path traversal issue
safe_dir = "/safe_dir/"
dir_name = os.path.dirname(args.input_meta)
if '../' in dir_name:
    sys.exit(1)
safe_dir = dir_name
file_name = os.path.basename(args.input_meta)
temp_dir = os.path.join(safe_dir, file_name)
with open(temp_dir, "r") as fi:
    out_file = temp_dir.split(".json")[0] + ".csv"
    with open(out_file, "w") as fo:
        for line in fi:
            try:
                obj = json.loads(line)
                cat = obj["categories"][0][-1]
                print(obj["asin"] + "\t" + cat, file=fo)
            except:
                print("Invalid line in input_meta file. Json like data is expected.")
