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
import copy
import multiprocessing
from typing import Any, List, Optional, Callable, Union, Dict

import torch
from torch import Tensor
from torch.multiprocessing.spawn import _wrap, ProcessContext
from torch.nn.parallel.distributed import DistributedDataParallel

import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.strategies.launchers import _SpawnLauncher
from pytorch_lightning.strategies import Strategy, DDPSpawnStrategy as _DDPSpawnStrategy
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.utilities.optimizer import optimizers_to_device
from pytorch_lightning.utilities.distributed import ReduceOp

from bigdl.nano.common.cpu_schedule import schedule_workers
from bigdl.nano.deps.ipex.ipex_api import ipex_device, ipex_optimize, \
    create_IPEXAccelerator_1_9, to_cpu
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10

import logging
import warnings

_STEP_OUTPUT_TYPE = Union[torch.Tensor, Dict[str, Any]]

log = logging.getLogger(__name__)


class _DDPSpawnLauncher(_SpawnLauncher):

    def __init__(self, strategy: Strategy) -> None:
        super().__init__(strategy)

    def launch(self, function: Callable, *args: Any,
               trainer: Optional["pl.Trainer"] = None, **kwargs: Any) -> Any:
        # pytorch_lightning 1.6 uses this method to create child processes
        os.environ["MASTER_PORT"] = str(self._strategy.cluster_environment.main_port)

        if self._strategy.cpu_for_each_process is None:
            cpu_procs = schedule_workers(self._strategy.num_processes)
        else:
            cpu_procs = self._strategy.cpu_for_each_process

        init_KMP_AFFINITY = os.environ.get("KMP_AFFINITY", "")
        init_OMP_NUM_THREADS = os.environ.get("OMP_NUM_THREADS", "")

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

        os.environ["KMP_AFFINITY"] = init_KMP_AFFINITY
        os.environ["OMP_NUM_THREADS"] = init_OMP_NUM_THREADS

        # recover the state of child process
        spawn_output = return_queue.get()
        self._recover_results_in_main_process(spawn_output, trainer)
        return spawn_output.trainer_results


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

    def _configure_launcher(self):
        self._launcher = _DDPSpawnLauncher(self)

    def setup(self, trainer: "pl.Trainer") -> None:
        # when using spawn, multiple child processes may update the weights of
        # the same model, so we should copy the model to avoid it
        if self.strategy_name == "ddp_spawn":
            self.model = copy.deepcopy(self.model)

        self.accelerator.setup(trainer)

        trainer_fn = trainer.state.fn
        if trainer_fn == TrainerFn.FITTING:
            if self._layer_sync:
                self.model = self._layer_sync.apply(self.model)

        self.setup_precision_plugin()

        # `configure_ddp` will create a `DistributedDataParallel`, which has no
        # `test_step` method in pytorch_lightning 1.6, which causes error when
        # calling `trainer.test()`, so we call `configure_ddp` only when fitting
        if trainer_fn == TrainerFn.FITTING:
            self.configure_ddp()
        else:
            # when calling `trainer.test()`, the `model.training` won't be set to `False`,
            # causing the following `ipex_optimize` to raise error,
            # so we need to set it to `False` manuallay
            self.model.training = False

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

        # some operations in `configure_ddp` do not support XPU,
        # which is used by ipex==1.9, so we move this line here
        self.model_to_device()

    def configure_ddp(self):
        # we should override this method to change the creation of `DistributedDataParallel`
        self.pre_configure_ddp()
        self._model = DistributedDataParallel(
            LightningDistributedModule(self.model),
            **self._ddp_kwargs,
        )
        self._register_ddp_hooks()

        self.setup_optimizers(self.lightning_module.trainer)
        optimizers_to_device(self.optimizers, self.root_device)

    def reduce(self, tensor, group: Optional[Any] = None,
               reduce_op: Union[ReduceOp, str] = "mean") -> Tensor:
        # some operations in `super.reduce()` method do not support XPU,
        # which will cause error when using ipex==1.9, however, these operations
        # seems not necessary, so we just ignore these errors
        try:
            return super().reduce(tensor, group, reduce_op)
        except Exception as _e:
            return tensor

    def training_step_end(self, output: _STEP_OUTPUT_TYPE) -> _STEP_OUTPUT_TYPE:
        """
        For ipex xpu tensor do not support `tensor.storage()` right now,
        which is a required operation by pytorch_lightning,
        so just move output to cpu to store it, and move it back when doing backward.
        """
        if self.use_ipex and TORCH_VERSION_LESS_1_10:
            output = to_cpu(output)

        return super().training_step_end(output)

    def test_step_end(self, output: Optional[_STEP_OUTPUT_TYPE]) -> \
            Optional[_STEP_OUTPUT_TYPE]:
        """A hook to do something at the end of the test step
        Args:
            output: the output of the test step
        """
        if self.use_ipex and TORCH_VERSION_LESS_1_10:
            output = to_cpu(output)

        return super().test_step_end(output)

    def validation_step_end(self, output: Optional[_STEP_OUTPUT_TYPE]) -> \
            Optional[_STEP_OUTPUT_TYPE]:
        """A hook to do something at the end of the validation step
        Args:
            output: the output of the validation step
        """
        if self.use_ipex and TORCH_VERSION_LESS_1_10:
            output = to_cpu(output)

        return super().validation_step_end(output)

    def backward(self,  # type: ignore
                 closure_loss: torch.Tensor,
                 *args,
                 **kwargs) -> torch.Tensor:
        """
        Moving back loss to xpu device
        """
        closure_loss = closure_loss.to(self.root_device)
        return super().backward(
            closure_loss,
            *args,
            **kwargs,
        )
