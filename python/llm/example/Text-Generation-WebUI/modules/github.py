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

# This file is adapted from
# https://github.com/oobabooga/text-generation-webui/blob/main/modules/github.py


import subprocess
from pathlib import Path

new_extensions = set()


def clone_or_pull_repository(github_url):
    global new_extensions

    repository_folder = Path("extensions")
    repo_name = github_url.rstrip("/").split("/")[-1].split(".")[0]

    # Check if the repository folder exists
    if not repository_folder.exists():
        repository_folder.mkdir(parents=True)

    repo_path = repository_folder / repo_name

    # Check if the repository is already cloned
    if repo_path.exists():
        yield f"Updating {github_url}..."
        # Perform a 'git pull' to update the repository
        try:
            pull_output = subprocess.check_output(["git", "-C", repo_path, "pull"], stderr=subprocess.STDOUT)
            yield "Done."
            return pull_output.decode()
        except subprocess.CalledProcessError as e:
            return str(e)

    # Clone the repository
    try:
        yield f"Cloning {github_url}..."
        clone_output = subprocess.check_output(["git", "clone", github_url, repo_path], stderr=subprocess.STDOUT)
        new_extensions.add(repo_name)
        yield f"The extension `{repo_name}` has been downloaded.\n\nPlease close the the web UI completely and launch it again to be able to load it."
        return clone_output.decode()
    except subprocess.CalledProcessError as e:
        return str(e)
