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

# this module includes path utils for the Huggingface repository
import os
from bigdl.nano.utils.common import invalidInputError


class NoPathException(Exception):
    pass


def get_local_path_from_repo_id(repo_id, models_root=os.getenv('HF_HOME')):
    # Applicable for diffusers models. Given a repo_id, get the local path of this model if exists

    if models_root is None:
        invalidInputError(
            False,
            errMsg="To use repo_id, you must set environmrnt variable `HF_HOME`.")

    repo_id, model_id = repo_id.split("/")
    # hardcode the diffusers path so that we only consider local models
    cache_dir = os.path.join(models_root, "diffusers", f"models--{repo_id}--{model_id}")
    model_path = get_snapshot_dir_from_cache_dir(cache_dir)
    return model_path


def get_snapshot_dir_from_cache_dir(cache_dir):
    # given a huggingface format cache dir, get the latest snapshot from it
    # TODO: probably add rolling strategy if any model fails
    invalidInputError(os.path.exists(cache_dir), errMsg=">> Local model does not exist.")
    snapshots_dir = os.path.join(cache_dir, "snapshots")
    snapshots = os.listdir(snapshots_dir)
    invalidInputError(
        len(snapshots) != 0,
        errMsg=f">> No models available, please download the model first")
    current_latest_snapshot = snapshots[0]
    current_latest_mtime = os.path.getmtime(os.path.join(snapshots_dir, current_latest_snapshot))
    for snap in snapshots:
        dir = os.path.join(snapshots_dir, snap)
        if os.path.getmtime(dir) > current_latest_mtime:
            current_latest_mtime = os.path.getmtime(dir)
            current_latest_snapshot = snap
    return os.path.join(snapshots_dir, current_latest_snapshot)
