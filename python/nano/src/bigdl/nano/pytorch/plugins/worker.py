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
import multiprocessing
from torch.multiprocessing.spawn import _wrap


if __name__ == '__main__':
    temp_dir = sys.argv[1]
    process_idx = int(os.environ["PROCESS_IDX"])

    # set the same `multiprocessing.current_process().authkey` as the main process
    # so that we can load the `args.pkl`
    authkey = bytes(os.environ['AUTHKEY'], encoding='utf-8')
    multiprocessing.current_process().authkey = authkey

    with open(os.path.join(temp_dir, "args.pkl"), "rb") as f:
        (fn, args, error_queue) = cloudpickle.load(f)

    _wrap(fn, process_idx, args, error_queue)
