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
    args = parser.parse_args()

    lines = json.load(open(args.input))
    ct_anony_votes = 0
    all_users = set()
    all_models = set()
    for l in lines:
        if not l["anony"]:
            continue
        all_users.add(l["judge"])
        all_models.add(l["model_a"])
        all_models.add(l["model_b"])
        ct_anony_votes += 1

    print(f"#anony_vote: {ct_anony_votes}, #user: {len(all_users)}")
    print(f"#model: {len(all_models)}")
