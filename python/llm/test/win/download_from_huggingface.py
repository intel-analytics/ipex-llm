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


import os
from huggingface_hub import login
from huggingface_hub import snapshot_download

if __name__ == '__main__':
    access_token_read = os.environ.get('HF_TOKEN')
    login(token = access_token_read)


    # Download bigscience/bloom-560m
    snapshot_download(repo_id="bigscience/bloom-560m", local_dir="models/bloom-560m",
                      local_dir_use_symlinks=False, ignore_patterns="*.safetensors")
    
    # # Download decapoda-research/llama-7b-hf
    # snapshot_download(repo_id="decapoda-research/llama-7b-hf", local_dir="models/llama-7b-hf",
    #                   local_dir_use_symlinks=False, ignore_patterns="*.safetensors")

    # Download togethercomputer/RedPajama-INCITE-Chat-3B-v1
    # snapshot_download(repo_id="togethercomputer/RedPajama-INCITE-Chat-3B-v1", local_dir="models/redpajama-3b",
    #                   local_dir_use_symlinks=False, ignore_patterns="*.safetensors")
