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
import requests

headers = {"authorization": "Bearer hf_XXX"}

url = "https://huggingface.co/api/datasets/lmsys/lmsys-chat-1m/user-access-request/pending"
a = requests.get(url, headers=headers)

for u in a.json():
    user = u["user"]["user"]
    url = "https://huggingface.co/api/datasets/lmsys/lmsys-chat-1m/user-access-request/grant"
    ret = requests.post(url, headers=headers, json={"user": user})
    print(user, ret.status_code)
    assert ret.status_code == 200
