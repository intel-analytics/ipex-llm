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
Split the dataset into training and test set.

Usage: python3 -m fastchat.data.split_train_test --in sharegpt.json
"""
import argparse
import json

import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--ratio", type=float, default=0.9)
    args = parser.parse_args()

    content = json.load(open(args.in_file, "r"))
    np.random.seed(0)

    perm = np.random.permutation(len(content))
    content = [content[i] for i in perm]
    split = int(args.ratio * len(content))

    train_set = content[:split]
    test_set = content[split:]

    print(f"#train: {len(train_set)}, #test: {len(test_set)}")
    train_name = args.in_file.replace(".json", "_train.json")
    test_name = args.in_file.replace(".json", "_test.json")
    json.dump(train_set, open(train_name, "w"), indent=2, ensure_ascii=False)
    json.dump(test_set, open(test_name, "w"), indent=2, ensure_ascii=False)
