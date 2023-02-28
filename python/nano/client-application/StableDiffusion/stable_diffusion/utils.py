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
import shutil
from huggingface_hub.constants import REPO_ID_SEPARATOR


model_version_map = {
    "v1.4": {"model_id": "CompVis/stable-diffusion-v1-4"},
    "v2.0-base": {"model_id": "stabilityai/stable-diffusion-2-base"},
    "v2.1-base": {"model_id": "stabilityai/stable-diffusion-2-1-base"}
}

def get_snapshot_dir_from_model_version(version):
    model_id = model_version_map[version]["model_id"]
    repo_id, model_version = model_id.split("/")
    model_dir = REPO_ID_SEPARATOR.join(["models", repo_id, model_version])
    snapshot_dir = os.path.join("models", model_dir, "snapshots")
    snapshots = os.listdir(snapshot_dir)
    assert len(snapshots) != 0, "No models available, please download the model first"
    latest_snapshot = snapshots[0]
    mtime = os.path.getmtime(os.path.join(snapshot_dir, snapshots[0]))
    # check if there are newer snapshots
    for snap in snapshots:
        dir = os.path.join(snapshot_dir, snap)
        if os.path.getmtime(dir) > mtime:
            mtime = os.path.getmtime(dir)
            latest_snapshot = snap
    
    return os.path.join(snapshot_dir, latest_snapshot)

def copy_model_from_snapshot(input, output):
    '''
    copy a huggingface snapshot to output folder
    '''
    for root, dirs, files in os.walk(input):
        for d in dirs:
            if not os.path.isdir(os.path.join(output, d)):
                os.mkdir(os.path.join(output, d))
        for f in files:
            # target_prefix = os.path.relpath(root, input)
            # shutil.copy2(os.path.abspath(os.path.join(root, f)), os.path.join(output, target_prefix), follow_symlinks=True)
            target_prefix = os.path.relpath(root, input)
            os.symlink(os.path.abspath(os.path.join(root, f)), os.path.join(output, target_prefix, f))
            
