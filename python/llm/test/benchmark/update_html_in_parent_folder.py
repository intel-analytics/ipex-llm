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
    
    current_folder = Path(folder_path)
    folder_list = [current_folder/'batch_size_1/',current_folder/'batch_size_2/',current_folder/'batch_size_4/',current_folder/'merged/']
    
    # List all html files under current folder and delete them
    for html_file in current_folder.glob('*.html'):
        html_file.unlink()
    for folder in folder_list:
        # Find latest html file under batch1/batch2/batch4/merged folders
        latest_html_file = max(Path(folder).glob('*.html'), key=os.path.getctime, default=None)
        # Copy the latest html file to parent folder
        if latest_html_file is not None:
            shutil.copy(latest_html_file, current_folder)
        print(latest_html_file.name)

def main():
    parser = argparse.ArgumentParser(description="Update HTML in parent folder.")
    parser.add_argument("-f", "--folder", type=str, help="Path to the folder")
    args = parser.parse_args()

    update_html_in_parent_folder(args.folder)

if __name__ == "__main__":
    main()
