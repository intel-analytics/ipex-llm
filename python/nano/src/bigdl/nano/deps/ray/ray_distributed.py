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
import pytorch_lightning as pl
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning.utilities.distributed import ReduceOp
from pytorch_lightning.strategies.launchers import _SpawnLauncher
from pytorch_lightning.strategies import DDPSpawnStrategy

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

        # when using pytorch lightning's trainer, the `trainer` cannot be None,
        # when using pytorch lightning's LightningLite, the `trainer` should be None
        if trainer is None:
            return results[0]

        # Get the results, checkpoint path, and model weights from worker 0.
        results, best_path, state_dict = results[0]  # type: ignore
        strategy.model.load_state_dict(state_dict)
        strategy.lightning_module.trainer = trainer
        strategy.lightning_module.trainer.checkpoint_callback.best_model_path = best_path

        return results.trainer_results  # type: ignore

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
        if trainer is None:
            return move_data_to_device(results, "cpu")

        results = strategy._launcher._collect_rank_zero_results(trainer, results)

        if strategy.global_rank == 0:
            if trainer.checkpoint_callback is not None:
                return move_data_to_device(results, "cpu"), \
                    trainer.checkpoint_callback.best_model_path, \
                    move_data_to_device(strategy.lightning_module.state_dict(), "cpu")
            else:
                return move_data_to_device(results, "cpu"), \
                    None, \
                    move_data_to_device(strategy.lightning_module.state_dict(), "cpu")


class RayStrategy(DDPSpawnStrategy):
    """A DDP Strategy which uses ray as backend."""

    strategy_name = "ray"

    def __init__(self,
                 num_workers: int = 1,
                 num_cpus_per_worker: int = 1,
                 use_gpu: bool = False,
                 use_ipex: bool = False,
                 enable_bf16: bool = False,
                 init_hook: Callable = None,
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
        self.enable_bf16 = enable_bf16

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
