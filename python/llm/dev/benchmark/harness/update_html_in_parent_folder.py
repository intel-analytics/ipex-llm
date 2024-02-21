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

# Python program to update Html in parent folder

import os
import shutil
import argparse
from pathlib import Path

def update_html_in_parent_folder(folder_path):
    # Get parent folder
    parent_folder = Path(folder_path).parent

    # List all html files under parent folder and delete them
    for html_file in parent_folder.glob('*.html'):
        html_file.unlink()

    # Find latest html file under folder_path
    latest_html_file = max(Path(folder_path).glob('*.html'), key=os.path.getctime, default=None)

    # Copy the latest html file to parent folder
    if latest_html_file is not None:
        shutil.copy(latest_html_file, parent_folder)
    
    print(latest_html_file.name)

def main():
    parser = argparse.ArgumentParser(description="Update HTML in parent folder.")
    parser.add_argument("-f", "--folder", type=str, help="Path to the folder")
    args = parser.parse_args()

    update_html_in_parent_folder(args.folder)

if __name__ == "__main__":
    main()
