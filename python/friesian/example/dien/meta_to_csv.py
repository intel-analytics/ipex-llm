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

parser = ArgumentParser()
parser.add_argument('--input_meta', type=str, required=True,
                    help="item metadata file")

args = parser.parse_args()
fi = open(args.input_meta, "r")
out_file = args.input_meta.split(".json")[0] + ".csv"
fo = open(out_file, "w")
for line in fi:
    try:
        obj = json.loads(line)
        cat = obj["categories"][0][-1]
        print(obj["asin"] + "\t" + cat, file=fo)
    except:
        print("Invalid line in input_meta file. Json like data is expected.")
