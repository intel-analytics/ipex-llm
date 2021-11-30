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

# This file is adapted from https://github.com/PyTorchLightning
# /pytorch-lightning/blob/master/pytorch_lightning/plugins/training_type/ddp_spawn.py
#
# Copyright The PyTorch Lightning team.
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


import os
from typing import Any, List, Optional, Callable

import multiprocessing
import torch
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.multiprocessing.spawn import _wrap, ProcessContext

import pytorch_lightning as pl
from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.seed import reset_seed

from bigdl.nano.common.cpu_schedule import schedule_workers
import logging
log = logging.getLogger(__name__)


def start_processes_new(fn, args=(), nprocs=1, join=True, daemon=False,
                        start_method='spawn', cpu_procs=None):
    mp = multiprocessing.get_context(start_method)
    error_queues = []
    processes = []

    if cpu_procs is None:
        cpu_procs = schedule_workers(nprocs)

    init_KMP_AFFINITY = os.environ.get("KMP_AFFINITY")
    init_OMP_NUM_THREADS = os.environ.get("OMP_NUM_THREADS")

    for i in range(nprocs):
        os.environ["KMP_AFFINITY"] = f"granularity=fine,proclist"\
                                     f"=[{','.join([str(i) for i in cpu_procs[i]])}],explicit"
        os.environ["OMP_NUM_THREADS"] = str(len(cpu_procs[i]))
        log.debug(f"[Process {i}]: using KMP_AFFINITY: {os.environ['KMP_AFFINITY']}")
        log.debug(f"[Process {i}]: using OMP_NUM_THREADS: {os.environ['OMP_NUM_THREADS']}")
        error_queue = mp.SimpleQueue()
        process = mp.Process(
            target=_wrap,
            args=(fn, i, args, error_queue),
            daemon=daemon,
        )
        process.start()
        error_queues.append(error_queue)
        processes.append(process)

    context = ProcessContext(processes, error_queues)
    if not join:
        return context

    # Loop on join until it returns True or raises an exception.
    while not context.join():
        pass

    os.environ["KMP_AFFINITY"] = init_KMP_AFFINITY
    os.environ["OMP_NUM_THREADS"] = init_OMP_NUM_THREADS


class DDPSpawnPlugin(pl.plugins.DDPSpawnPlugin):
    distributed_backend = "ddp_spawn"

    def __init__(
        self,
        parallel_devices: Optional[List[torch.device]] = None,
        num_nodes: int = 1,
        cluster_environment: ClusterEnvironment = None,
        sync_batchnorm: bool = False,
        ddp_comm_state: Optional[object] = None,
        ddp_comm_hook: Optional[Callable] = None,
        ddp_comm_wrapper: Optional[Callable] = None,
        cpu_for_each_process: Optional[List[List[int]]] = None,
        **kwargs: Any,
    ):
        super().__init__(parallel_devices,
                         num_nodes,
                         cluster_environment,
                         sync_batchnorm,
                         ddp_comm_state,
                         ddp_comm_hook,
                         ddp_comm_wrapper,
                         **kwargs)
        self.cpu_for_each_process = cpu_for_each_process
        self.is_distributed = True

    @property
    def mp_spawn_kwargs(self):
        return {
            "args": (self.lightning_module.trainer, self.mp_queue),
            "nprocs": self.num_processes,
            "cpu_procs": self.cpu_for_each_process
        }

    def start_training(self, trainer):
        # reset ortsess, since InferenceSession can not be pickled
        self.model._ortsess = None
        start_processes_new(self.new_process, **self.mp_spawn_kwargs)
        # reset optimizers, since main process is never used for training
        # and thus does not have a valid optim state
        trainer.optimizers = []

    def start_evaluating(self, trainer):
        print("evaluate")
        start_processes_new(self.new_process, **self.mp_spawn_kwargs)

    def start_predicting(self, trainer):
        print("predict")
        start_processes_new(self.new_process, **self.mp_spawn_kwargs)

    def new_process(self, process_idx, trainer, mp_queue):
        self.mp_queue = mp_queue

        reset_seed()

        self.set_world_ranks(process_idx)

        # set warning rank
        rank_zero_only.rank = self.global_rank

        # set up server using proc 0's ip address
        # try to init for 20 times at max in case ports are taken
        # where to store ip_table
        self.init_ddp_connection(self.global_rank, self.world_size)

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

    def configure_ddp(self):
        self.pre_configure_ddp()
        self._model = DistributedDataParallel(
            LightningDistributedModule(self.model),
            **self._ddp_kwargs,
        )
        self._register_ddp_hooks()
