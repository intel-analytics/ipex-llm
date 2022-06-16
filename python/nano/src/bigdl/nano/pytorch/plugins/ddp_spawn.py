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
import multiprocessing
from typing import Any, List, Optional, Callable

import torch
from torch.multiprocessing.spawn import _wrap, ProcessContext

import pytorch_lightning as pl
from pytorch_lightning.strategies.launchers import _SpawnLauncher
from pytorch_lightning.strategies import Strategy, DDPSpawnStrategy as _DDPSpawnStrategy
from pytorch_lightning.plugins.environments import LightningEnvironment

from bigdl.nano.common.cpu_schedule import schedule_workers
from bigdl.nano.deps.ipex.ipex_api import ipex_device, ipex_optimize, create_IPEXAccelerator_1_9
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10

import logging
import warnings
import copy
log = logging.getLogger(__name__)

class _DDPSpawnLauncher(_SpawnLauncher):
    r"""Spawns processes that run a given function in parallel, and joins them all at the end.

    The main process in which this launcher is invoked creates N so-called worker processes (using
    :func:`torch.multiprocessing.spawn`) that run the given function.
    Worker processes have a rank that ranges from 0 to N - 1.

    Note:
        - This launcher requires all objects to be pickleable.
        - It is important that the entry point to the program/script is guarded by ``if __name__ == "__main__"``.

    Args:
        strategy: A reference to the strategy that is used together with this launcher.
    """

    def __init__(self, strategy: Strategy) -> None:
        super().__init__(strategy)

    def launch(self, function: Callable, *args: Any, trainer: Optional["pl.Trainer"] = None, **kwargs: Any) -> Any:
        """Spawns processes that run the given function in parallel.

        The function is allowed to have a return value. However, when all processes join, only the return value
        of worker process 0 gets returned from this `launch` method in the main process.

        Arguments:
            function: The entry point for all spawned processes.
            *args: Optional positional arguments to be passed to the given function.
            trainer: Optional reference to the :class:`~pytorch_lightning.trainer.trainer.Trainer` for which
                a selected set of attributes get restored in the main process after processes join.
            **kwargs: Optional keyword arguments to be passed to the given function.
        """
        # The default cluster environment in Lightning chooses a random free port number
        # This needs to be done in the main process here before spawning to ensure each rank will connect
        # through the same port
        os.environ["MASTER_PORT"] = str(self._strategy.cluster_environment.main_port)

        if self._strategy.cpu_for_each_process is None:
            cpu_procs = schedule_workers(self._strategy.num_processes)
        else:
            cpu_procs = self._strategy.cpu_for_each_process
        
        init_KMP_AFFINITY = os.environ.get("KMP_AFFINITY")
        init_OMP_NUM_THREADS = os.environ.get("OMP_NUM_THREADS")

        mp = multiprocessing.get_context(self._start_method)
        return_queue = mp.SimpleQueue()
        error_queues = []
        processes = []
        args = (trainer, function, args, kwargs, return_queue)

        for i in range(self._strategy.num_processes):
            os.environ["KMP_AFFINITY"] = f"granularity=fine,proclist"\
                                         f"=[{','.join([str(i) for i in cpu_procs[i]])}],explicit"
            os.environ["OMP_NUM_THREADS"] = str(len(cpu_procs[i]))
            log.debug(f"[Process {i}]: using KMP_AFFINITY: {os.environ['KMP_AFFINITY']}")
            log.debug(f"[Process {i}]: using OMP_NUM_THREADS: {os.environ['OMP_NUM_THREADS']}")
            error_queue = mp.SimpleQueue()
            process = mp.Process(
                target=_wrap,
                args=(self._wrapping_function, i, args, error_queue),
                daemon=False,
            )
            process.start()
            error_queues.append(error_queue)
            processes.append(process)

        context = ProcessContext(processes, error_queues)

        while not context.join():
            pass

        if init_KMP_AFFINITY is not None:
            os.environ["KMP_AFFINITY"] = init_KMP_AFFINITY
        if init_OMP_NUM_THREADS is not None:
            os.environ["OMP_NUM_THREADS"] = init_OMP_NUM_THREADS


class DDPSpawnStrategy(_DDPSpawnStrategy):

    strategy_name = "ddp_spawn"

    def __init__(
        self,
        num_processes: int = 1,
        cpu_for_each_process: Optional[List[List[int]]] = None,
        use_ipex=False,
        enable_bf16=False,
        **kwargs: Any
    ):
        device = ipex_device() if use_ipex and TORCH_VERSION_LESS_1_10 else 'cpu'
        parallel_devices = [torch.device(device) for _ in range(num_processes)]
        cluster_environment = LightningEnvironment()

        if use_ipex and TORCH_VERSION_LESS_1_10 and 'accelerator' not in kwargs:
            super().__init__(accelerator=create_IPEXAccelerator_1_9(),
                             parallel_devices=parallel_devices,
                             cluster_environment=cluster_environment, **kwargs)
        else:
            super().__init__(parallel_devices=parallel_devices,
                             cluster_environment=cluster_environment, **kwargs)
        self.cpu_for_each_process = cpu_for_each_process
        self.is_distributed = True
        self.use_ipex = use_ipex
        self.enable_bf16 = enable_bf16

    def setup(self, trainer: "pl.Trainer") -> None:
        super().setup(trainer)

        if self.use_ipex and not TORCH_VERSION_LESS_1_10:
            dtype = torch.bfloat16 if self.enable_bf16 else None
            num_optimizers = len(self.optimizers)

            if num_optimizers == 1:
                optimizer = self.optimizers[0]
                ipex_optimize(self.model, optimizer=optimizer, inplace=True, dtype=dtype)
            elif num_optimizers == 0:
                ipex_optimize(self.model, inplace=True, dtype=dtype)
            else:
                warnings.warn(f"IPEX currently only support single optimizers, "
                              f"but got {num_optimizers}. Skip IPEX")

    def _configure_launcher(self):
        self._launcher = _DDPSpawnLauncher(self)
