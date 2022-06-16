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

import cloudpickle
import multiprocessing
import os
import subprocess
import sys
import copy
from typing import Any, List, Optional, Callable
from tempfile import TemporaryDirectory

import torch

import pytorch_lightning as pl
from pytorch_lightning.strategies.launchers import _Launcher
from pytorch_lightning.strategies import Strategy, DDPSpawnStrategy
from pytorch_lightning.plugins.environments import LightningEnvironment

from bigdl.nano.common.cpu_schedule import schedule_workers
from bigdl.nano.deps.ipex.ipex_api import ipex_device, ipex_optimize, create_IPEXAccelerator, create_IPEXAccelerator_1_9
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10

import logging

log = logging.getLogger(__name__)


def queue_dumper(q):
    """Copy values from Queue into a List."""
    q_list = []
    while not q.empty():
        q_list.append(q.get())
    return q_list


def queue_loader(q_list):
    """Copy values from a List into a SimpleQueue."""
    q = multiprocessing.SimpleQueue()
    for item in q_list:
        q.put(item)
    return q


class _DDPSubprocessLauncher(_Launcher):
    def __init__(self, strategy: Strategy) -> None:
        self._strategy = strategy

    @property
    def is_interactive_compatible(self) -> bool:
        """Returns whether this launcher can work in interactive environments such as Jupyter notebooks."""
        return False

    def launch(self, function: Callable, *args: Any, trainer: Optional["pl.Trainer"] = None, **kwargs: Any) -> Any:
        """Creates new processes, then calls the given function.

        Arguments:
            function: A callback function to execute after all processes have been created.
                It is up to the implementation of this function to synchronize the processes, e.g., with barriers.
            *args: Optional positional arguments to be passed to the given function.
            trainer: Optional reference to the :class:`~pytorch_lightning.trainer.trainer.Trainer`.
            **kwargs: Optional keyword arguments to be passed to the given function.
        """
        os.environ["MASTER_PORT"] = str(self._strategy.cluster_environment.main_port)

        strategy = self._strategy
        cpu_procs = schedule_workers(strategy.num_processes)

        if strategy.is_global_zero and not torch.distributed.is_initialized():
            log.info("-" * 100)
            log.info(f"distributed_backend={strategy.distributed_backend}")
            log.info(f"All DDP processes registered. Starting ddp with {strategy.world_size} processes")
            log.info("-" * 100)

        with TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, "args.pkl"), 'wb') as f:
                args = strategy
                cloudpickle.dump(args, f)

            processes = []
            cwd_path = os.path.split(os.path.realpath(__file__))[0]
            for i in range(strategy.num_processes):

                env = copy.deepcopy(os.environ)

                env.update({
                    "KMP_AFFINITY": f"granularity=fine,proclist"
                                    f"=[{','.join([str(i) for i in cpu_procs[i]])}],explicit",
                    "OMP_NUM_THREADS": str(len(cpu_procs[i])),
                    "PROCESS_IDX": str(i),
                })

                processes.append(subprocess.Popen([sys.executable, f"{cwd_path}/worker.py",
                                                temp_dir], env=env))

            for _, process in enumerate(processes):
                process.wait()

            for _, process in enumerate(processes):
                assert process.returncode == 0, "Subprocess incorrectly exit, \
                                                check the trainer configure or usage"

            with open(os.path.join(temp_dir, "results.pkl"), "rb") as f:
                queue_list = cloudpickle.load(f)
                strategy.mp_queue = queue_loader(queue_list)


class DDPSubprocessStrategy(DDPSpawnStrategy):

    distributed_backend = "ddp_subprocess"

    def __init__(
        self,
        num_processes: int = 1,
        cpu_for_each_process: Optional[List[List[int]]] = None,
        use_ipex=False,
        enable_bf16=False,
        **kwargs: Any
    ):
        """Create a DDPSpawnStrategy, adding a cpu_for_each_process parameter."""
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
        self.use_ipex = use_ipex
        self.enable_bf16 = enable_bf16

    def _configure_launcher(self):
        self._launcher = _DDPSubprocessLauncher(self)




# class DDPSubprocessPlugin(DDPSpawnPlugin):
#     """
#     Extending DDPSpawnPlugin to support launch subprocesses with optimized env variables.

#     Instead of using python multiprocessing.spawn, this plugin use subprocess.Popen to start
#     a new process in order to run mulit-instance training in a jupyter notebook.
#     """

#     distributed_backend = "ddp_subprocess"

#     def _run_subprocess(self, tmpdir):
#         from bigdl.nano.common.cpu_schedule import schedule_workers

#         cpu_procs = schedule_workers(self.num_processes)

#         processes = []
#         cwd_path = os.path.split(os.path.realpath(__file__))[0]
#         for i in range(self.num_processes):

#             env = copy.deepcopy(os.environ)

#             env.update({
#                 "KMP_AFFINITY": f"granularity=fine,proclist"
#                                 f"=[{','.join([str(i) for i in cpu_procs[i]])}],explicit",
#                 "OMP_NUM_THREADS": str(len(cpu_procs[i])),
#                 "PROCESS_IDX": str(i),
#             })

#             processes.append(subprocess.Popen([sys.executable, f"{cwd_path}/worker.py",
#                                                tmpdir], env=env))

#         return processes

#     def start_training(self, trainer):
#         """Run training in multiple processes."""
#         self.model._ortsess = None
#         self._execution_loop()
#         trainer.optimizers = []

#     def start_evaluating(self, trainer):
#         """Run evaluating in multiple processes."""
#         print("evaluate")
#         self._execution_loop()

#     def start_predicting(self, trainer):
#         """Run predicting in multiple processes."""
#         print("predict")
#         self._execution_loop()

#     def _execution_loop(self):
#         if self.is_global_zero and not torch.distributed.is_initialized():
#             log.info("-" * 100)
#             log.info(f"distributed_backend={self.distributed_backend}")
#             log.info(f"All DDP processes registered. Starting ddp with {self.world_size} processes")
#             log.info("-" * 100)

#         with TemporaryDirectory() as temp_dir:
#             with open(os.path.join(temp_dir, "args.pkl"), 'wb') as f:
#                 args = self
#                 cloudpickle.dump(args, f)

#             processes = self._run_subprocess(temp_dir)

#             for _, process in enumerate(processes):
#                 process.wait()

#             for _, process in enumerate(processes):
#                 assert process.returncode == 0, "Subprocess incorrectly exit, \
#                                                 check the trainer configure or usage"

#             with open(os.path.join(temp_dir, "results.pkl"), "rb") as f:
#                 queue_list = cloudpickle.load(f)
#                 self.mp_queue = queue_loader(queue_list)
