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

if __name__ == '__main__':
    # Set number of threads in subprocess
    tf.config.threading.set_inter_op_parallelism_threads(int(sys.argv[2]))
    tf.config.threading.set_intra_op_parallelism_threads(int(sys.argv[2]))
    temp_dir = sys.argv[1]

    with open(os.path.join(temp_dir, "args.pkl"), 'rb') as f:
        args = cloudpickle.load(f)

    with open(os.path.join(temp_dir, "target.pkl"), 'rb') as f:
        target = cloudpickle.load(f)

    history = target(*args)
    tf_config = json.loads(os.environ["TF_CONFIG"])

    with open(os.path.join(temp_dir,
                           f"history_{tf_config['task']['index']}"), "wb") as f:
        cloudpickle.dump(history, f)
