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

import json
import os
import cloudpickle
import sys
import tensorflow as tf

safe_dir = '/home/'

def is_path_within_safe_directory(requested_path):
    # Append '/' to the safe directory to account for the check when requested_path is the same as safe_dir.
    safe_dir_with_slash = safe_dir if safe_dir.endswith('/') else safe_dir + '/'
    return os.path.commonprefix((os.path.realpath(requested_path), safe_dir_with_slash)) == safe_dir_with_slash

def main():
    # Set number of threads in subprocess
    tf.config.threading.set_inter_op_parallelism_threads(int(sys.argv[2]))
    tf.config.threading.set_intra_op_parallelism_threads(int(sys.argv[2]))
    temp_dir = sys.argv[1]

    if not is_path_within_safe_directory(temp_dir):
        print("Bad user! The requested path is not allowed.")
        return

    with open(os.path.join(temp_dir, "args.pkl"), 'rb') as f:
        args = cloudpickle.load(f)

    with open(os.path.join(temp_dir, "target.pkl"), 'rb') as f:
        target = cloudpickle.load(f)

    history = target(*args)
    tf_config = json.loads(os.environ["TF_CONFIG"])

    with open(os.path.join(temp_dir,
                           f"history_{tf_config['task']['index']}"), "wb") as f:
        cloudpickle.dump(history, f)

if __name__ == '__main__':
    main()
