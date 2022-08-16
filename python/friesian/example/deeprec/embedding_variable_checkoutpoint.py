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

# Original imports
from tensorflow.contrib.framework.python.framework import checkpoint_utils
import argparse

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
                        help='Path to the model checkpoint.',
                        type=str,
                        required=False)
    return parser

parser = get_arg_parser()
args = parser.parse_args()
checkpoint_dir = args.checkpoint

for name, shape in checkpoint_utils.list_variables(checkpoint_dir):
     print('loading...', name, shape, checkpoint_utils.load_variable(checkpoint_dir,name))