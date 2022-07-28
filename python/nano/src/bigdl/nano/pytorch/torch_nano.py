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

from typing import Any
from logging import warning
from functools import partial
from abc import abstractmethod

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from pytorch_lightning.lite import LightningLite
from pytorch_lightning.lite.wrappers import _LiteModule, _LiteOptimizer

from bigdl.nano.common import check_avx512
from bigdl.nano.utils.log4Error import invalidInputError
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10, TORCH_VERSION_LESS_1_11
from bigdl.nano.pytorch.strategies.ipex.ipex_api import ipex_optimize
from bigdl.nano.pytorch.strategies import create_IPEXStrategy, DDPSpawnStrategy, \
    DDPSubprocessStrategy, create_RayStrategy


class TorchNano(LightningLite):
    """
    TorchNano for BigDL-Nano pytorch.

    It can be used to accelerate custom pytorch training loops with very few code changes.
    """

    def __init__(self, num_processes: int = 1,
                 use_ipex: bool = False,
                 enable_bf16: bool = False,
                 strategy: str = "subprocess",
                 *args, **kwargs) -> None:
        """
        Create a TorchNano with nano acceleration.

        :param num_processes: number of processes in distributed training, defaults to 1
        :param use_ipex: whether use ipex acceleration, defaults to False
        :param enable_bf16: whether use bf16 acceleration, defaults to False
        :param strategy: use which backend in distributed mode, defaults to "subprocess", \
            now avaiable strategies are 'spawn', 'subprocess' and 'ray'
        """
        self.num_processes = num_processes
        self.use_ipex = use_ipex
        self.enable_bf16 = enable_bf16

        if TORCH_VERSION_LESS_1_11 and use_ipex and not check_avx512():
            warning("Enable ipex<=1.10 in a cpu instruction set"
                    " without avx512 will crash."
                    "Fall back to regular pytorch.")
            self.use_ipex = False

        if self.num_processes == 1:
            if self.use_ipex:
                strategy = create_IPEXStrategy(enable_bf16=self.enable_bf16)
            else:
                strategy = None     # type: ignore
        elif strategy == "spawn":
            strategy = DDPSpawnStrategy(num_processes=self.num_processes,   # type: ignore
                                        use_ipex=self.use_ipex,
                                        enable_bf16=self.enable_bf16)
        elif strategy == "subprocess":
            strategy = DDPSubprocessStrategy(num_processes=self.num_processes,  # type: ignore
                                             use_ipex=self.use_ipex,
                                             enable_bf16=self.enable_bf16)
        elif strategy == "ray":
            strategy = create_RayStrategy(num_workers=self.num_processes,
                                          use_ipex=self.use_ipex,
                                          enable_bf16=self.enable_bf16)
        else:
            warning(f"Bigdl-nano doesn't support '{strategy}' strategy now, "
                    f"'{strategy}' strategy of pytorch_lightning will be used. "
                    f"Supported strategies are 'spawn', 'subprocess' and 'ray'.")

        kwargs["strategy"] = strategy
        super().__init__(*args, **kwargs)

        setattr(self, "train", partial(self._run_impl, self.train))

    def _setup(
        self,
        model: nn.Module,
        *optimizers: Optimizer,
        move_to_device: bool = True,
    ) -> Any:
        """Used to replace LightningLite's setup method."""
        # LightningLite won't call `Strategy.setup()` method,
        # in which we add IPEX's optimization when using `trainer`.

        # When we call `TorchNano().train()`, it will call
        # `Strategy.setup_environment()` -> `Lanucher.launch()` -> user defined `train()` method.

        # However the model and optimizers haven't been specified when calling these three methods,
        # so we have to add optimizations in this method, which will be called in
        # user defined `train()` method.

        # add IPEX 1.11's optimization
        if self.use_ipex and not TORCH_VERSION_LESS_1_10:
            dtype = torch.bfloat16 if self.enable_bf16 else None
            if len(optimizers) == 0:
                ipex_optimize(model, inplace=True, dtype=dtype)
            elif len(optimizers) == 1:
                ipex_optimize(model, optimizer=optimizers[0], inplace=True, dtype=dtype)
            else:
                invalidInputError(False, "Ipex does not support more than one optimizers.")

        # the following codes are copied from pl's LightningLite's `setup` method,
        # ipex 1.9 requires `_move_model_to_device` after `_setup_model_and_optimizers`, but
        # pl's `setup` method calls `_move_model_to_device` before `_setup_model_and_optimizers`,
        # so we copy the codes and swap their order.
        self._validate_setup(model, optimizers)

        model, optimizers = self._strategy._setup_model_and_optimizers(model, list(optimizers))
        if move_to_device:
            model = self._move_model_to_device(model=model, optimizers=list(optimizers))
        model = _LiteModule(model, self._precision_plugin)
        optimizers = [_LiteOptimizer(optimizer=optimizer, strategy=self._strategy)  # type: ignore
                      for optimizer in optimizers]
        self._models_setup += 1
        if optimizers:
            # join both types in a list for API convenience
            return [model] + optimizers  # type: ignore
        return model

    def setup(self, model: nn.Module, optimizer: Optimizer,     # type: ignore[override]
              *dataloaders: DataLoader, move_to_device: bool = True):
        """
        Setup model, optimizer, loss function and dataloaders for accelerated training.

        :param model: A model to setup
        :param optimizer: The optimizer to setup
        :param *dataloaders: The dataloader(s) to setup
        :param move_to_device: If set ``True`` (default), moves the model to the correct device. \
            Set this to ``False`` and alternatively use :meth:`to_device` manually.
        :return: The tuple of the wrapped model, optimizer, loss_func and dataloaders, \
            in the same order they were passed in.
        """
        model, optimizer = self._setup(model, optimizer, move_to_device=move_to_device)
        dataloaders = self.setup_dataloaders(*dataloaders,  # type: ignore
                                             move_to_device=move_to_device)
        return model, optimizer, dataloaders

    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> Any:
        """
        All the code inside this train method gets accelerated by TorchNano.

        You can pass arbitrary arguments to this function when overriding it.
        """

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Only for compatibility, don't use it."""
        # this is a abstract method, so we must implement it
        pass
