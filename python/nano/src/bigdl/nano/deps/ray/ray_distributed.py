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

# This file is adapted from ray_lightning. https://github.com/ray-project/ray_lightning
# Copyright 2021 Ray Lightning Author
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
import warnings
from typing import Callable, Dict, List, Union, Any, Optional
from collections import defaultdict

import ray
from ray.util.ml_utils.util import find_free_port

import torch
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler
import pytorch_lightning as pl
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning.utilities.distributed import ReduceOp
from pytorch_lightning.strategies.launchers import _SpawnLauncher
from pytorch_lightning.strategies import DDPSpawnStrategy
from pytorch_lightning.core.optimizer import _configure_schedulers_automatic_opt
from pytorch_lightning.core.optimizer import _configure_schedulers_manual_opt
from pytorch_lightning.core.optimizer import _set_scheduler_opt_idx, _validate_scheduler_api
from pytorch_lightning.core.optimizer import LightningOptimizer

from .ray_envbase import RayEnvironment
from bigdl.nano.utils.log4Error import invalidInputError
from bigdl.nano.pytorch.strategies.ipex.ipex_api import ipex_device, ipex_optimize, \
    create_IPEXAccelerator, to_cpu
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10

_STEP_OUTPUT_TYPE = Union[torch.Tensor, Dict[str, Any]]


@ray.remote     # type: ignore
class RayExecutor:
    """A class to execute any arbitrary function remotely."""

    def set_env_var(self, key: str, value: str):
        """Set an environment variable with the provided values."""
        if value is not None:
            value = str(value)
            os.environ[key] = value

    def set_env_vars(self, keys: List[str], values: List[str]):
        """Sets multiple env vars with the provided values."""
        invalidInputError(len(keys) == len(values),
                          "keys length doesn't mathcc values length")
        for key, value in zip(keys, values):
            self.set_env_var(key, value)

    def get_env_vars(self, key: str):
        """Return the specified environment variable."""
        return os.environ[key]

    def get_node_ip(self):
        """Returns the IP address of the node that this Ray actor is on."""
        return ray.util.get_node_ip_address()

    def execute(self, fn: Callable, *args, **kwargs):
        """Execute the provided function and return the result."""
        return fn(*args, **kwargs)


class _RayLauncher(_SpawnLauncher):

    def __init__(self, strategy: 'RayStrategy') -> None:
        self._strategy: RayStrategy = strategy

    @property
    def is_interactive_compatible(self) -> bool:
        return False

    def launch(self, function: Callable, *args: Any,
               trainer: Optional["pl.Trainer"] = None, **kwargs: Any) -> Any:
        # pytorch_lightning 1.6 uses this method to create child processes
        strategy = self._strategy

        # fix bug, see ddp_spawn strategy for details
        if strategy.use_ipex and TORCH_VERSION_LESS_1_10 and trainer is not None:
            if isinstance(args[1], LightningDataModule):
                args[1].trainer = None
            elif isinstance(args[3], LightningDataModule):
                args[3].trainer = None
            elif isinstance(args[4], LightningDataModule):
                args[4].trainer = None

        strategy._setup_env_vars()
        strategy.global_to_local = strategy.get_local_ranks()   # type: ignore

        torch_backend = os.getenv("PL_TORCH_DISTRIBUTED_BACKEND")
        if torch_backend is None:
            torch_backend = "nccl" if strategy.use_gpu else "gloo"
        strategy._process_group_backend = torch_backend

        futures = [
            strategy.workers[i].execute.remote(
                self._wrapping_function,
                (i, trainer, strategy, function, args, kwargs)
            )
            for i in range(strategy.num_workers)
        ]

        results = ray.get(futures)  # type: ignore
        ray.shutdown()  # release the resources occupied by ray

        # when using pytorch lightning's trainer, the `trainer` cannot be None,
        # when using pytorch lightning's LightningLite, the `trainer` should be None
        if trainer is None:
            return results[0]

        self._recover_results_in_main_process(results[0], trainer)

        return results[0].trainer_results

    @staticmethod
    def _wrapping_function(args_pack: tuple) -> Any:   # type: ignore[override]
        global_rank, trainer, strategy, function, args, kwargs = args_pack
        invalidInputError(isinstance(strategy, RayStrategy), "expect ray strategy here")
        invalidInputError(isinstance(strategy.cluster_environment, RayEnvironment),
                          "expect ray environment here")

        strategy.cluster_environment.set_global_rank(global_rank)
        strategy.cluster_environment.set_remote_execution(True)

        strategy._worker_setup(global_rank)
        results = function(*args, **kwargs)

        # when using pytorch lightning's trainer, the `trainer` cannot be None,
        # when using pytorch lightning's LightningLite, the `trainer` should be None
        if trainer is not None:
            results = strategy._launcher._collect_rank_zero_results(trainer, results)

        if strategy.global_rank == 0:
            return move_data_to_device(results, "cpu")


class RayStrategy(DDPSpawnStrategy):
    """A DDP Strategy which uses ray as backend."""

    strategy_name = "ray"

    def __init__(self,
                 num_workers: int = 1,
                 num_cpus_per_worker: int = 1,
                 use_gpu: bool = False,
                 use_ipex: bool = False,
                 dtype=None,
                 init_hook: Callable = None,
                 auto_lr: Union[bool, dict] = True,
                 **ddp_kwargs: Any):
        """Create a RayStrategy."""
        # Unset MKL setting as bigdl.nano would give default values when init env.
        # Running different programs may need different configurations.
        # refer to https://analytics-zoo-doc.readthedocs.io/en/master/zoo.ray.html#
        # zoo.ray.raycontext.RayServiceFuncGenerator
        # Also KMP_AFFINITY may entangle something doing with ray._raylet.CoreWorker,
        # which is invoked by ray.init and leads to stocking.
        os.environ.pop("KMP_AFFINITY", None)
        os.environ.pop("OMP_NUM_THREADS", None)

        if not ray.is_initialized():    # type: ignore
            print(ray.init())   # type: ignore

        if use_ipex and TORCH_VERSION_LESS_1_10 and 'accelerator' not in ddp_kwargs:
            super().__init__(accelerator=create_IPEXAccelerator(),
                             parallel_devices=[],
                             cluster_environment=RayEnvironment(world_size=num_workers),
                             **ddp_kwargs)
        else:
            super().__init__(parallel_devices=[],
                             cluster_environment=RayEnvironment(world_size=num_workers),
                             **ddp_kwargs)

        self.num_workers = num_workers
        self.num_cpus_per_worker = num_cpus_per_worker
        self.use_gpu = use_gpu
        self.use_ipex = use_ipex
        self.dtype = dtype
        self.auto_lr = auto_lr

        invalidInputError(not self.use_gpu or not self.use_ipex,
                          "You can not specify gpu and ipex at the same time.")

        self.workers = self._create_worker()
        self.init_hook = init_hook
        self._local_rank = 0

        if self.init_hook:
            ray.get([w.execute.remote(self.init_hook) for w in self.workers])   # type: ignore

    def _configure_launcher(self):
        self._launcher = _RayLauncher(self)

    def _create_worker(self):
        """Creates Ray actor."""
        from bigdl.nano.common.cpu_schedule import schedule_processors

        envs = schedule_processors(self.num_workers)

        workers = []
        for i in range(self.num_workers):
            worker = RayExecutor.options(
                num_cpus=self.num_cpus_per_worker,
                num_gpus=int(self.use_gpu)
            ).remote()

            ray.get(worker.set_env_var.remote("KMP_AFFINITY", envs[i]['KMP_AFFINITY']))
            ray.get(worker.set_env_var.remote("OMP_NUM_THREADS", envs[i]['OMP_NUM_THREADS']))
            # pytest will set this environment variable to the path of `nano` directory,
            # and subprocess will use it to initialize its `sys.path`,
            # so we can import `test` module in subprocess
            ray.get(worker.set_env_var.remote("PYTHONPATH", envs[i].get("PYTHONPATH", "")))

            workers.append(worker)

        return workers

    def _setup_env_vars(self):
        # Get rank 0 worker address and port for DDP connection.
        os.environ["MASTER_ADDR"] = ray.get(
            self.workers[0].get_node_ip.remote())
        os.environ["MASTER_PORT"] = str(
            ray.get(self.workers[0].execute.remote(find_free_port)))

        # Set environment variables for remote workers.
        keys = [
            "PL_GLOBAL_SEED", "PL_TORCH_DISTRIBUTED_BACKEND",
            "MASTER_ADDR", "MASTER_PORT"
        ]
        values = [os.getenv(k) for k in keys]
        ray.get([w.set_env_vars.remote(keys, values) for w in self.workers])

    def get_local_ranks(self):
        """Creates a mapping of global ranks to local ranks."""
        # Get the local ranks for all the workers and store as a dict.
        # First get the IP address of each remote worker.
        node_ips = ray.get([w.get_node_ip.remote() for w in self.workers])
        rank_counter_dict = defaultdict(int)  # type: ignore
        global_to_local = [None] * self.num_workers
        for global_rank in range(self.num_workers):
            ip = node_ips[global_rank]
            global_to_local[global_rank] = rank_counter_dict[ip]
            rank_counter_dict[ip] += 1
        return global_to_local

    def set_world_ranks(self, process_idx: int = 0):
        """Set the appropriate rank attribues for the trainer."""
        invalidInputError(self.cluster_environment is not None and isinstance(
            self.cluster_environment, RayEnvironment), "expect ray environment here")
        if self.cluster_environment.is_remote():    # type: ignore
            self._local_rank = self.global_to_local[self.global_rank]   # type: ignore
            self.cluster_environment.set_global_rank(self.global_rank)
            self.cluster_environment.set_world_size(self.num_workers)
            rank_zero_only.rank = self.cluster_environment.global_rank()  # type: ignore

    def setup(self, trainer: "pl.Trainer") -> None:
        """Setup the distributed environment of ray executor, we add ipex optimization here."""
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

    @property
    def root_device(self):
        """Return the root device."""
        if self.use_ipex and TORCH_VERSION_LESS_1_10:
            # Add ipex option.
            return torch.device(ipex_device())
        else:
            return torch.device("cpu")

    def determine_ddp_device_ids(self):
        """Return the index of root device."""
        # For ipex case, we also should not return any optional device id.
        if self.root_device.type == "cpu" or self.root_device.type == "xpu":
            return None
        return [self.root_device.index]

    @property
    def distributed_sampler_kwargs(self):
        """Returns the args to use for torch.data.DistributedSampler."""
        distributed_sampler_kwargs = dict(
            num_replicas=self.num_workers, rank=self.global_rank)
        return distributed_sampler_kwargs

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

    def on_train_start(self) -> None:
        """Setup warmup lr_schedulers after resetting the train dataloaders."""
        # LightnigModule.train_dataloader() generate the training dataloaders after setup,
        # so config the warmup lr_schedulers in on_train_start hook to infer warmup_steps.
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

    def backward(self,  # type: ignore
                 closure_loss: torch.Tensor,
                 *args,
                 **kwargs) -> torch.Tensor:
        """Moving back loss to xpu device."""
        closure_loss = closure_loss.to(self.root_device)
        return super().backward(
            closure_loss,
            *args,
            **kwargs,
        )
