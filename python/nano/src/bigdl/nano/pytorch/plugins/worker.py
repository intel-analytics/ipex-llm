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

import torch
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.seed import reset_seed


import logging
log = logging.getLogger(__name__)



def new_process(self, process_idx, trainer, mp_queue):


    # TODO: we moved it to the trainer.fit after calling pre_dispatch
    #   ... need to double check that it is the correct place
    # self.trainer.call_setup_hook(self.model)

    # on world_size=0 let everyone know training is starting
    if self.is_global_zero and not torch.distributed.is_initialized():
        log.info("-" * 100)
        log.info(f"distributed_backend={self.distributed_backend}")
        log.info(f"All DDP processes registered. Starting ddp with {self.world_size} processes")
        log.info("-" * 100)

    # set the ranks and devices
    self.dist.rank = self.global_rank
    self.dist.device = self.root_device

    if self.sync_batchnorm:
        self.model = self.configure_sync_batchnorm(self.model)

    self.configure_ddp()

    # Move this line here so that we can temporarily use cpu while configuring ddp
    # and use ipex.DEVICE later on
    # move the model to the correct device
    self.model_to_device()

    self.barrier()
    results = trainer.run_stage()

    # persist info in ddp_spawn
    self.transfer_distrib_spawn_state_on_fit_end(results)


if __name__ == '__main__':
    temp_dir = sys.argv[1]

    with open(os.path.join(temp_dir, "args.pkl"), 'rb') as f:
        args = pickle.load(f)

    plugin, trainer, mp_queue = args
    process_idx = os.environ["process_idx"]

    reset_seed()
    plugin.set_world_ranks(process_idx)
    rank_zero_only.rank = plugin.global_rank

    plugin.init_ddp_connection(plugin.global_rank, plugin.world_size)

    if plugin.is_global_zero and not torch.distributed.is_initialized():
        log.info("-" * 100)
        log.info(f"distributed_backend={plugin.distributed_backend}")
        log.info(f"All DDP processes registered. Starting ddp with {plugin.world_size} processes")
        log.info("-" * 100)

    plugin.dist.rank = plugin.global_rank
    plugin.dist.device = plugin.root_device

    if plugin.sync_batchnorm:
        plugin.model = plugin.configure_sync_batchnorm(plugin.model)

    plugin.configure_ddp()

    # Move this line here so that we can temporarily use cpu while configuring ddp
    # and use ipex.DEVICE later on
    # move the model to the correct device
    plugin.model_to_device()

    plugin.barrier()
    results = trainer.run_stage()

    # persist info in ddp_spawn
    plugin.transfer_distrib_spawn_state_on_fit_end(results)

    if plugin.global_rank == 0 and plugin.mp_queue is not None:
        with open(os.path.join(temp_dir,
                               "results.pkl"), "wb") as f:
            cloudpickle.dump(plugin.mp_queue, f)
