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
import json
import multiprocessing

import torch
import deepspeed
import deepspeed.comm as dist
import cloudpickle

from bigdl.nano.pytorch import InferenceOptimizer
from bigdl.nano.pytorch.dispatcher import patch_torch


if __name__ == '__main__':
    temp_dir = sys.argv[1]

    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    # set the same `multiprocessing.current_process().authkey` as the main process
    # so that we can load the `sys_path.pkl` and `args.pkl`
    authkey = bytes(os.environ['AUTHKEY'], encoding='utf-8')
    multiprocessing.current_process().authkey = authkey

    # restore main process's sys.path to avoid potential bugs when loading `args.pkl`
    # i.e. cannot find some modules located in main process's sys.path
    with open(os.path.join(temp_dir, "sys_path.json"), "r") as f:
        sys.path = json.load(f)

    with open(os.path.join(temp_dir, "patch_status.json"), "r") as f:
        patch_status = json.load(f)
        if patch_status['patch_torch']:
            patch_torch(cuda_to_cpu=patch_status['patch_cuda'])

    model = torch.load(os.path.join(temp_dir, "model.bin")).eval()

    # todo: use broadcast instread of passing tensor by recv_queue
    with open(os.path.join(temp_dir, "queues.pkl"), "rb") as f:
        recv_queue, send_queue = cloudpickle.load(f)

    # tell parent process that `temp_dir` can be deleted
    # todo: a better way to do this
    send_queue.put(None)

    # todo: support bf16
    model = deepspeed.init_inference(model, mp_size=world_size,
                                     dtype=torch.float32, replace_with_kernel_inject=True)

    # todo: error handling
    with InferenceOptimizer.get_context(model):
        while True:
            args = recv_queue.get()
            if world_size > 1:
                dist.barrier()
            output = model(*args)
            if local_rank == 0:
                send_queue.put(output)
