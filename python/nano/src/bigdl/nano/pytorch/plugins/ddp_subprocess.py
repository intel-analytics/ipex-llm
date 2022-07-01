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
import os
import multiprocessing
import subprocess
import sys
import copy
import uuid
from typing import Any, Optional, Callable
from tempfile import TemporaryDirectory

import pytorch_lightning as pl
from pytorch_lightning.core.datamodule import LightningDataModule

from bigdl.nano.pytorch.plugins.ddp_spawn import DDPSpawnStrategy, _DDPSpawnLauncher
from bigdl.nano.common.cpu_schedule import schedule_workers
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10
from bigdl.nano.utils.log4Error import invalidInputError

import logging

log = logging.getLogger(__name__)


class _DDPSubprocessLauncher(_DDPSpawnLauncher):

    @property
    def is_interactive_compatible(self) -> bool:
        return True

    def launch(self, function: Callable, *args: Any,
               trainer: Optional["pl.Trainer"] = None, **kwargs: Any) -> Any:
        # pytorch_lightning 1.6 uses this method to create child processes
        invalidInputError(trainer is not None and self._strategy.cluster_environment is not None,
                          'strategy.cluster_environment and trainer cannot be None')

        os.environ["MASTER_PORT"] = str(self._strategy.cluster_environment.main_port)

        if self._strategy.cpu_for_each_process is None:
            cpu_procs = schedule_workers(self._strategy.num_processes)
        else:
            cpu_procs = self._strategy.cpu_for_each_process

        # fix bug
        # args[1] is dataloader, args[3] and args[4] is datamodule
        if self._strategy.use_ipex and TORCH_VERSION_LESS_1_10:
            if isinstance(args[1], LightningDataModule):
                args[1].trainer = None
            elif isinstance(args[3], LightningDataModule):
                args[3].trainer = None
            elif isinstance(args[4], LightningDataModule):
                args[4].trainer = None

        # the `return_queue` is necessary for recovering child process's state, we need
        # to dump it in this process and load it in subprocess, the `mp.SimpleQueue()` in
        # `ddp_spawn.py` cannot be dumped, so we use `multiprocessing.Manager().Queue()` here,
        # however, in order to be able to load it in subprocess, we must ensure the
        # `current_process().authkey` in this process and in subprocess are the same
        authkey = str(uuid.uuid1())
        multiprocessing.current_process().authkey = bytes(authkey, encoding='utf-8')
        mp = multiprocessing.Manager()

        with TemporaryDirectory() as temp_dir:
            return_queue = mp.Queue()
            error_queue = mp.Queue()
            args = (trainer, function, args, kwargs, return_queue)
            with open(os.path.join(temp_dir, "args.pkl"), "wb") as f:
                cloudpickle.dump((self._wrapping_function, args, error_queue), f)

            processes = []
            cwd_path = os.path.split(os.path.realpath(__file__))[0]
            for i in range(self._strategy.num_processes):
                env = copy.deepcopy(os.environ)

                env.update({
                    "KMP_AFFINITY": f"granularity=fine,proclist"
                                    f"=[{','.join([str(i) for i in cpu_procs[i]])}],explicit",
                    "OMP_NUM_THREADS": str(len(cpu_procs[i])),
                    "PROCESS_IDX": str(i),
                    "AUTHKEY": authkey,
                })
                log.debug(f"[Process {i}]: using KMP_AFFINITY: {env['KMP_AFFINITY']}")
                log.debug(f"[Process {i}]: using OMP_NUM_THREADS: {env['OMP_NUM_THREADS']}")

                processes.append(subprocess.Popen([sys.executable, f"{cwd_path}/worker.py",
                                                   temp_dir], env=env))

            for _, process in enumerate(processes):
                process.wait()

            for _, process in enumerate(processes):
                assert process.returncode == 0, "Subprocess incorrectly exit, \
                                                check the trainer configure or usage"
            # recover the state of child process
            spawn_output = return_queue.get()
            self._recover_results_in_main_process(spawn_output, trainer)
            return spawn_output.trainer_results


class DDPSubprocessStrategy(DDPSpawnStrategy):
    """
    Extending DDPSpawnStrategy to support launch subprocesses with optimized env variables.

    Instead of using python multiprocessing.spawn, this strategy use subprocess.Popen to start
    a new process in order to run mulit-instance training in a jupyter notebook.
    """

    strategy_name = "ddp_subprocess"

    def _configure_launcher(self):
        self._launcher = _DDPSubprocessLauncher(self)
