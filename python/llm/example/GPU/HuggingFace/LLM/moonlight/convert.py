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
import argparse
from safetensors.torch import load_file, save_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Moonlight model to be sucessfully loaded by transformers')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the downloaded Moonlight model')

    args = parser.parse_args()
    model_path = args.model_path
    converted_model_path = model_path + '-converted'

    if os.path.exists(converted_model_path):
        shutil.rmtree(converted_model_path)

    os.makedirs(converted_model_path)

    for f in os.listdir(model_path):
        f_path = os.path.join(model_path, f)
        f_dst_path = os.path.join(converted_model_path, f)

        print(f)

        if f.endswith(".safetensors"):
            save_file(load_file(f_path), f_dst_path, metadata={"format": "pt"})
        elif not f.startswith(".") and os.path.isfile(f_path): # skip file/dir name start with .
            shutil.copyfile(f_path, f_dst_path)

    print(f"Converted model successfully saved to {converted_model_path}")
