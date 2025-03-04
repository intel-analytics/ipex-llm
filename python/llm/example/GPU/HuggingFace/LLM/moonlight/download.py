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

import argparse
from huggingface_hub import snapshot_download

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Moonlight model to be sucessfully loaded by transformers')
    parser.add_argument('--repo-id', type=str, default='moonshotai/Moonlight-16B-A3B-Instruct',
                        help='Hugging Face model repo id to download')
    parser.add_argument('--commit-id', type=str, required=True,
                        help='Revision of the downloaded model')
    parser.add_argument('--download-dir-path', type=str,
                        help='Folder path where the model will be downloaded')

    args = parser.parse_args()

    repo_id = args.repo_id
    download_dir_path = args.download_dir_path
    if download_dir_path is None:
        download_dir_path = './' + repo_id.rsplit("/", 1)[-1]

    snapshot_download(repo_id=repo_id,
                      revision=args.commit_id,
                      local_dir=download_dir_path)

    print(f'{repo_id} has been downloaded to {download_dir_path}')
