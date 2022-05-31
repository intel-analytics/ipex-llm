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

import multiprocessing
import os
import sys

import cloudpickle

from pytorch_lightning.utilities.seed import reset_seed

from bigdl.nano.pytorch.plugins.ddp_subprocess import queue_dumper
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10
from bigdl.nano.deps.ipex.ipex_api import ipex_device, ipex_optimize
import torch
import warnings


if __name__ == '__main__':
    temp_dir = sys.argv[1]

    with open(os.path.join(temp_dir, "args.pkl"), 'rb') as f:
        args = cloudpickle.load(f)

    plugin = args
    trainer = plugin.lightning_module.trainer
    plugin.mp_queue = multiprocessing.SimpleQueue()
    process_idx = int(os.environ["PROCESS_IDX"])

    reset_seed()
    plugin.set_world_ranks(process_idx)
    # rank_zero_only.rank = plugin.global_rank

    plugin.init_ddp_connection(plugin.global_rank, plugin.world_size)

    plugin.dist.rank = plugin.global_rank
    plugin.dist.device = plugin.root_device

    if plugin.use_ipex and not TORCH_VERSION_LESS_1_10:
        dtype = torch.bfloat16 if plugin.enable_bf16 else None
        num_optimizers = len(plugin.lightning_module.trainer.accelerator.optimizers)
        if num_optimizers == 1:
            optimizer = plugin.lightning_module.trainer.accelerator.optimizers[0]
            ipex_optimize(plugin.model, optimizer=optimizer,
                          inplace=True, dtype=dtype)
        elif num_optimizers == 0:
            plugin.model.eval()
            ipex_optimize(plugin.model, inplace=True, dtype=dtype)
        else:
            warnings.warn(f"IPEX currently only support single optimizers, "
                          f"but got {num_optimizers}. Skip IPEX")

    if plugin.sync_batchnorm:
        plugin.model = plugin.configure_sync_batchnorm(plugin.model)

    plugin.configure_ddp()

    plugin.model_to_device()

    plugin.barrier()
    results = trainer.run_stage()

    plugin.transfer_distrib_spawn_state_on_fit_end(results)
    if plugin.global_rank == 0:
        with open(os.path.join(temp_dir,
                               "results.pkl"), "wb") as f:
            results = queue_dumper(plugin.mp_queue)
            cloudpickle.dump(results, f)
