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
from torch import nn
from torch import Tensor
from torch.multiprocessing.spawn import _wrap, ProcessContext
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim.lr_scheduler import _LRScheduler

import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.core.optimizer import _configure_schedulers_automatic_opt
from pytorch_lightning.core.optimizer import _configure_schedulers_manual_opt
from pytorch_lightning.core.optimizer import _set_scheduler_opt_idx, _validate_scheduler_api
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.strategies.launchers import _SpawnLauncher
from pytorch_lightning.strategies import DDPSpawnStrategy as _DDPSpawnStrategy
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.utilities.distributed import ReduceOp

from bigdl.nano.common.cpu_schedule import schedule_processors
from bigdl.nano.pytorch.strategies.ipex.ipex_api import ipex_device, \
    ipex_optimize, create_IPEXAccelerator, to_cpu
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10
from bigdl.nano.utils.log4Error import invalidInputError

import logging
import warnings

_STEP_OUTPUT_TYPE = Union[torch.Tensor, Dict[str, Any]]

log = logging.getLogger(__name__)


class _DDPSpawnLauncher(_SpawnLauncher):

    def __init__(self, strategy: 'DDPSpawnStrategy') -> None:   # type: ignore[override]
        self._strategy: DDPSpawnStrategy = strategy
        self._start_method = "spawn"

    def launch(self, function: Callable, *args: Any,
               trainer: Optional["pl.Trainer"] = None, **kwargs: Any) -> Any:
        # pytorch_lightning 1.6 uses this method to create child processes

        # the `self._strategy.cluster_environment` should not be None in normal circumstances,
        # if you see this error message, please report an issue in BigDL.
        invalidInputError(self._strategy.cluster_environment is not None,
                          'strategy.cluster_environment cannot be None')

        os.environ["MASTER_PORT"] = str(self._strategy.cluster_environment.main_port)

        cpu_procs = self._strategy.cpu_for_each_process
        if cpu_procs is None:
            envs = schedule_processors(self._strategy.num_processes)
        else:
            envs = [{
                "KMP_AFFINITY": f"granularity=fine,proclist"
                                f"=[{','.join([str(i) for i in cpu_procs[i]])}],explicit",
                "OMP_NUM_THREADS": str(len(cpu_procs[i]))
            } for i in range(self._strategy.num_processes)]

        # reset datamodule to fix bug:
        # in pytorch lightning 1.6, `datamodule` has a `trainer` member,
        # if this datamodule has been used for training, its `trainer` member will refers
        # to the previous trainer, which will causes errors when creating child processes.
        # pytorch lightning 1.4 resets datamodule automatically before creating child processes,
        # so we do not need to do this, but 1.6 resets datamodule after creating child processes,
        # so we must reset datamodule here.
        if self._strategy.use_ipex and TORCH_VERSION_LESS_1_10 and trainer is not None:
            # args[1] is dataloader, args[3] is datamodule when training,
            # and args[4] is datamodule when testing
            if isinstance(args[1], LightningDataModule):
                args[1].trainer = None
            elif isinstance(args[3], LightningDataModule):
                args[3].trainer = None
            elif isinstance(args[4], LightningDataModule):
                args[4].trainer = None

        init_KMP_AFFINITY = os.environ.get("KMP_AFFINITY", "")
        init_OMP_NUM_THREADS = os.environ.get("OMP_NUM_THREADS", "")

        mp = multiprocessing.get_context(self._start_method)
        return_queue = mp.SimpleQueue()
        error_queues = []
        processes = []
        args = (trainer, function, args, kwargs, return_queue)

        for i in range(self._strategy.num_processes):
            os.environ["KMP_AFFINITY"] = envs[i]['KMP_AFFINITY']
            os.environ["OMP_NUM_THREADS"] = envs[i]['OMP_NUM_THREADS']
            log.debug(f"[Process {i}]: using KMP_AFFINITY: {os.environ['KMP_AFFINITY']}")
            log.debug(f"[Process {i}]: using OMP_NUM_THREADS: {os.environ['OMP_NUM_THREADS']}")
            error_queue = mp.SimpleQueue()
            process = mp.Process(   # type: ignore
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

        # restore the state of child process
        spawn_output = return_queue.get()

        # when using pytorch lightning's trainer, the `trainer` cannot be None,
        # when using pytorch lightning's LightningLite, the `trainer` should be None
        if trainer is None:
            return spawn_output

        self._recover_results_in_main_process(spawn_output, trainer)
        return spawn_output.trainer_results


class DDPSpawnStrategy(_DDPSpawnStrategy):
    """Extending DDPSpawnStrategy to support launch subprocesses with optimized env variables."""

    strategy_name = "ddp_spawn"

    def __init__(
        self,
        num_processes: int = 1,
        cpu_for_each_process: Optional[List[List[int]]] = None,
        use_ipex=False,
        dtype=None,
        auto_lr=False,
        **kwargs: Any
    ):
        """Create a DDPSpawnStrategy, adding a cpu_for_each_process parameter."""
        device = ipex_device() if use_ipex and TORCH_VERSION_LESS_1_10 else 'cpu'
        parallel_devices = [torch.device(device) for _ in range(num_processes)]
        cluster_environment = LightningEnvironment()

        if use_ipex and TORCH_VERSION_LESS_1_10 and 'accelerator' not in kwargs:
            super().__init__(accelerator=create_IPEXAccelerator(),
                             parallel_devices=parallel_devices,
                             cluster_environment=cluster_environment,
                             **kwargs)
            if dtype == torch.bfloat16:
                import intel_pytorch_extension as ipex
                # Automatically mix precision
                ipex.enable_auto_mixed_precision(mixed_dtype=torch.bfloat16)
        elif use_ipex and dtype == torch.bfloat16 and 'precision_plugin' not in kwargs:
            from bigdl.nano.pytorch.strategies.ipex.ipex_strategy import IPEXBF16Precision
            super().__init__(parallel_devices=parallel_devices,
                             cluster_environment=cluster_environment,
                             precision_plugin=IPEXBF16Precision(), **kwargs)
        else:
            super().__init__(parallel_devices=parallel_devices,
                             cluster_environment=cluster_environment, **kwargs)
        self.cpu_for_each_process = cpu_for_each_process
        self.is_distributed = True
        self.use_ipex = use_ipex
        self.dtype = dtype
        self.auto_lr = auto_lr

    def _configure_launcher(self):
        self._launcher = _DDPSpawnLauncher(self)

    def setup(self, trainer: "pl.Trainer") -> None:
        """Setup the distributed environment of sub processes, we add ipex optimization here."""
        invalidInputError(self.model is not None, "You must specify the model.")

        # when using spawn, multiple child processes may update the weights of
        # the same model, so we should copy the model to avoid it
        if self.strategy_name == "ddp_spawn":
            # in pl 1.6, a trainer holds a  a strategy holds a model, a model holds a trainer,
            # `trainer.model` equals to `trainer.strategy.model`, so after assigning to
            # `self.model`, the trainer's model will refer to new model automatically
            self.model = copy.deepcopy(self.model)
            # `copy.deepcopy(self.model)` can't copy `self.model.trainer` correctly sometimes,
            # so we reuse the original trainer
            self.model.trainer = trainer    # type: ignore

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
            # `trainer.test()` won't set `model.training` to `False` automatically in pl 1.6,
            # then the following `ipex_optimize()` call will report an error,
            # so we need to set it to `False` manuallay
            self.model.eval()

        if trainer.training and self.auto_lr:

            def _unpack_lightning_optimizer(opt):
                return opt._optimizer if isinstance(opt, LightningOptimizer) else opt

            optimizers = self.optimizers
            optimizers = [_unpack_lightning_optimizer(opt) for opt in optimizers]

            for optimizer in optimizers:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= self.world_size

            lr_scheduler_configs = self.lr_scheduler_configs
            for config in lr_scheduler_configs:
                scheduler = config.scheduler
                if isinstance(scheduler, _LRScheduler):
                    scheduler.base_lrs = [  # type: ignore
                        lr * self.world_size for lr in scheduler.base_lrs  # type: ignore
                    ]

        if self.use_ipex and not TORCH_VERSION_LESS_1_10:
            num_optimizers = len(self.optimizers)

            if num_optimizers == 1:
                optimizer = self.optimizers[0]
                ipex_optimize(self.model, optimizer=optimizer, inplace=True, dtype=self.dtype)
            elif num_optimizers == 0:
                ipex_optimize(self.model, inplace=True, dtype=self.dtype)
            else:
                warnings.warn(f"IPEX currently only support single optimizers, "
                              f"but got {num_optimizers}. Skip IPEX")

        # some operations in `configure_ddp` do not support XPU,
        # which is used by ipex==1.9, so we move this line here
        self.model_to_device()

    def on_train_start(self):
        """Setup warmup lr_schedulers after resetting the train dataloaders."""
        # LightnigModule.train_dataloader() generate the training dataloaders after setup,
        # so attach the warmup lr_schedulers in on_train_start hook to infer warmup_steps.
        if not self.auto_lr:
            return
        if self.lr_scheduler_configs:
            warnings.warn(f"Nano warmup currently only support no scheduler, "
                          f"but got {len(self.lr_scheduler_configs)}. Skip warmup")
        else:
            trainer = self.lightning_module.trainer
            lr_schedulers = []
            warmup_params = {
                'start_factor': 1.0 / self.world_size,
                'end_factor': 1.0,
                'warmup_epochs': trainer.max_epochs // 10,
                'interval': 'epoch'
            }
            supported_keys = {'warmup_epochs'}
            if isinstance(self.auto_lr, dict):
                extra_keys = self.auto_lr.keys() - supported_keys
                if extra_keys:
                    warnings.warn(f"Found unsupported keys in the auto_lr dict: {extra_keys}")
                if 'warmup_epochs' not in self.auto_lr:
                    self.auto_lr = True
                    warnings.warn("Not found \"warmup_epochs\" in the auto_lr dict"
                                  " warmup_epochs is set by default")
                else:
                    invalidInputError(type(self.auto_lr['warmup_epochs']) is int,
                                      f"\"warmup_epochs\" is {type(self.auto_lr['warmup_epochs'])}",
                                      "expect \"warmup_epochs\" is a integer")
                    warmup_params['warmup_epochs'] = self.auto_lr['warmup_epochs']
            if type(self.auto_lr) is bool:
                # Call scheduler.step() after each minibatch rather than epoch if max_epochs < 10
                if warmup_params['warmup_epochs'] == 0:
                    train_loader = trainer.train_dataloader
                    max_steps = len(train_loader) * trainer.max_epochs
                    warmup_params['warmup_epochs'] = max_steps // 10
                    warmup_params['interval'] = 'step'
            for opt_idx, opt in enumerate(self.optimizers):
                from torch.optim.lr_scheduler import LambdaLR

                def lr_func(epoch):
                    current_epoch = trainer.current_epoch
                    start_factor = warmup_params['start_factor']
                    end_factor = warmup_params['end_factor']
                    total_iters = warmup_params['warmup_epochs']
                    if current_epoch > 0 and warmup_params['interval'] == 'step' \
                            or epoch > total_iters:
                        return 1.0
                    if epoch == 0:
                        return start_factor
                    return (end_factor - start_factor) * epoch / total_iters \
                        + start_factor
                scheduler = LambdaLR(optimizer=opt,
                                     lr_lambda=[lr_func] * len(opt.param_groups))
                lr_scheduler = {
                    'scheduler': scheduler,
                    'opt_idx': opt_idx,
                    'interval': warmup_params['interval']
                }
                lr_schedulers.append(lr_scheduler)

            # validates the lr_scheduler_configs, adapted from lightning
            # https://github.com/Lightning-AI/lightning/blob/1.6.4/pytorch_lightning/core/optimizer.py#L175
            lr_scheduler_configs = (
                _configure_schedulers_automatic_opt(lr_schedulers, None)
                if self.lightning_module.automatic_optimization
                else _configure_schedulers_manual_opt(lr_schedulers)
            )
            _set_scheduler_opt_idx(self.optimizers, lr_scheduler_configs)
            _validate_scheduler_api(lr_scheduler_configs, self.lightning_module)
            self.lr_scheduler_configs = lr_scheduler_configs

    def _setup_model(self, model: nn.Module) -> DistributedDataParallel:
        """Wraps the model into a 'DistributedDataParallel' module."""
        # we should override this method to change the creation of `DistributedDataParallel`
        return DistributedDataParallel(model, **self._ddp_kwargs)

    def reduce(self, tensor, group: Optional[Any] = None,   # type: ignore[override]
               reduce_op: Union[ReduceOp, str] = "mean") -> Tensor:
        """Reduces a tensor from several distributed processes to one aggregated tensor."""
        # some operations in `super.reduce()` method do not support XPU,
        # which will cause error when using ipex==1.9, however, these operations
        # seems not necessary, so we just ignore these errors
        try:
            return super().reduce(tensor, group, reduce_op)
        except Exception as _e:
            return tensor

    def training_step_end(self, output: _STEP_OUTPUT_TYPE) -> _STEP_OUTPUT_TYPE:
        """
        A hook to do something at the end of the train step.

        For ipex xpu tensor do not support `tensor.storage()` right now,
        which is a required operation by pytorch_lightning,
        so just move output to cpu to store it, and move it back when doing backward.
        """
        if self.use_ipex and TORCH_VERSION_LESS_1_10:
            output = to_cpu(output)

        return super().training_step_end(output)

    def test_step_end(self, output: Optional[_STEP_OUTPUT_TYPE]) -> \
            Optional[_STEP_OUTPUT_TYPE]:
        """
        A hook to do something at the end of the test step.

        :param output: the output of the test step
        """
        if self.use_ipex and TORCH_VERSION_LESS_1_10:
            output = to_cpu(output)

        return super().test_step_end(output)

    def validation_step_end(self, output: Optional[_STEP_OUTPUT_TYPE]) -> \
            Optional[_STEP_OUTPUT_TYPE]:
        """
        A hook to do something at the end of the validation step.

        :param output: the output of the validation step
        """
        if self.use_ipex and TORCH_VERSION_LESS_1_10:
            output = to_cpu(output)

        return super().validation_step_end(output)

    def backward(self, closure_loss: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Moving back loss to xpu device."""
        closure_loss = closure_loss.to(self.root_device)
        return super().backward(closure_loss, *args, **kwargs)
