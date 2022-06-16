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
import sys

import cloudpickle


if __name__ == '__main__':
    temp_dir = sys.argv[1]

    with open(os.path.join(temp_dir, "strategy.pkl"), 'rb') as f:
        strategy = cloudpickle.load(f)
    with open(os.path.join(temp_dir, "args.pkl"), "rb") as f:
        (args, kwargs) = cloudpickle.load(f)
    with open(os.path.join(temp_dir, "function.pkl"), 'rb') as f:
        function = cloudpickle.load(f)

    process_idx = int(os.environ["PROCESS_IDX"])

    strategy._worker_setup(process_idx)

    results = function(*args, **kwargs)

    if strategy.global_rank == 0:
        with open(os.path.join(temp_dir, "results.pkl"), "wb") as f:
            cloudpickle.dump(results, f)
