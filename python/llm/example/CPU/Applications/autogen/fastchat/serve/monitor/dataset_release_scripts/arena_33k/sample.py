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
Count the unique users in a battle log file.

Usage:
python3 -input in.json --number 1000
"""

import argparse
import json
import random

K = 1000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--number", type=int, nargs="+")
    args = parser.parse_args()

    convs = json.load(open(args.input))
    random.seed(0)
    random.shuffle(convs)

    for number in args.number:
        new_convs = convs[:number]

        output = args.input.replace(".json", f"_{number//K}k.json")
        with open(output, "w") as fout:
            json.dump(new_convs, fout, indent=2, ensure_ascii=False)

        print(f"#in: {len(convs)}, #out: {len(new_convs)}")
        print(f"Write to file: {output}")
