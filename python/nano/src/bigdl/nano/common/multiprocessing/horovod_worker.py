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
import cloudpickle
import sys


if __name__ == '__main__':
    temp_dir = sys.argv[1]

    with open(os.path.join(temp_dir, "args.pkl"), 'rb') as f:
        args = cloudpickle.load(f)

    with open(os.path.join(temp_dir, "target.pkl"), 'rb') as f:
        target = cloudpickle.load(f)

    import horovod.tensorflow.keras as hvd
    hvd.init()

    history = target(*args)
    idx = hvd.rank()

    with open(os.path.join(temp_dir,
                           f"history_{idx}"), "wb") as f:
        cloudpickle.dump(history, f)
