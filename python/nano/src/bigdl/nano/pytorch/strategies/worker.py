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
from bigdl.nano.pytorch.dispatcher import patch_torch


if __name__ == '__main__':
    temp_dir = sys.argv[1]
    process_idx = int(os.environ["PROCESS_IDX"])

    # set the same `multiprocessing.current_process().authkey` as the main process
    # so that we can load the `sys_path.pkl` and `args.pkl`
    authkey = bytes(os.environ['AUTHKEY'], encoding='utf-8')
    multiprocessing.current_process().authkey = authkey

    # restore main process's sys.path to avoid potential bugs when loading `args.pkl`
    # i.e. cannot find some modules located in main process's sys.path
    with open(os.path.join(temp_dir, "sys_path.pkl"), "rb") as f:
        sys.path = cloudpickle.load(f)

    with open(os.path.join(temp_dir, "patch_status.pkl"), "rb") as f:
        patch_status = cloudpickle.load(f)
        if patch_status['patch_torch']:
            patch_torch(cuda_to_cpu=patch_status['patch_cuda'])

    with open(os.path.join(temp_dir, "args.pkl"), "rb") as f:
        (fn, args, error_queue) = cloudpickle.load(f)

    # args[0] is `trainer`, when it is None, it means when are using LightningLite,
    # otherwise we are using trainer, for the details here, see `ddp_subprocess.py`
    if args[0] is None:
        _wrap(fn, process_idx, args, error_queue)
    else:
        _wrap(args[0].strategy._launcher._wrapping_function, process_idx, args, error_queue)
