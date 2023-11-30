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
Manually register workers.

Usage:
python3 -m fastchat.serve.register_worker --controller http://localhost:21001 --worker-name http://localhost:21002
"""

import argparse

import requests

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-address", type=str)
    parser.add_argument("--worker-name", type=str)
    parser.add_argument("--check-heart-beat", action="store_true")
    args = parser.parse_args()

    url = args.controller_address + "/register_worker"
    data = {
        "worker_name": args.worker_name,
        "check_heart_beat": args.check_heart_beat,
        "worker_status": None,
    }
    r = requests.post(url, json=data)
    assert r.status_code == 200
