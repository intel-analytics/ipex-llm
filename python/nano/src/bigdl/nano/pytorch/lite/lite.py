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

from typing import Any, Union
from logging import warning

import torch
from torch import nn
from torch.optim import Optimizer
import pytorch_lightning.lite as lite
from pytorch_lightning.strategies import Strategy
from pytorch_lightning.lite.wrappers import _LiteModule, _LiteOptimizer

from bigdl.nano.utils.log4Error import invalidInputError
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10
from bigdl.nano.pytorch.strategies.ipex.ipex_api import ipex_optimize
from bigdl.nano.pytorch.strategies import create_IPEXStrategy, DDPSpawnStrategy, \
    DDPSubprocessStrategy, create_RayStrategy


class LightningLite(lite.LightningLite):
    """
    LightningLite for BigDL-Nano pytorch.

    This LightningLite extends PyTorch Lightning's LightningLite by adding
    various options to accelerate pytorch training.
    """

    def __init__(self, num_processes: int = 1,
                 use_ipex: bool = False,
                 enable_bf16: bool = False,
                 strategy: Union[str, Strategy] = "subprocess",
                 *args, **kwargs) -> None:
        """
        Create a LightningLite with nano acceleration.

        :param num_processes: number of processes in distributed training, defaults to 1
        :param use_ipex: whether use ipex acceleration, defaults to False
        :param enable_bf16: whether use bf16 acceleration, defaults to False
        :param strategy: use which backend in distributed mode, defaults to "subprocess"
        """
        # Check arguments
        invalidInputError(isinstance(strategy, str),
                          "strategy object will be created by bigdl-nano, "
                          "you should pass the name of strategy.")

        self.num_processes = num_processes
        self.use_ipex = use_ipex
        self.enable_bf16 = enable_bf16

        if self.num_processes == 1:
            if self.use_ipex:
                strategy = create_IPEXStrategy(enable_bf16=self.enable_bf16)
            else:
                strategy = None     # type: ignore
        elif strategy == "spawn":
            strategy = DDPSpawnStrategy(num_processes=self.num_processes,
                                        use_ipex=self.use_ipex,
                                        enable_bf16=self.enable_bf16)
        elif strategy == "subprocess":
            strategy = DDPSubprocessStrategy(num_processes=self.num_processes,
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

    def setup(
        self,
        model: nn.Module,
        *optimizers: Optimizer,
        move_to_device: bool = True,
    ) -> Any:
        """
        Setup a model and its optimizers for accelerated training.

        LightningLite won't call `Strategy.setup()` method,
        in which we add IPEX's optimization when using `trainer`.

        When we call `LightningLite().run()`, it will call
        `Strategy.setup_environment()` -> `Lanucher.launch()` -> user defined `run()` method.

        However the model and optimizers haven't been specified when calling these three methods,
        so we have to add optimizations in this method, which will be called in
        user defined `run()` method.

        :param model: A model to setup
        :param *optimizers: The optimizer(s) to setup (no optimizers is also possible)
        :param move_to_device: If set ``True`` (default), moves the model to the correct device.
            Set this to ``False`` and alternatively use :meth:`to_device` manually.
        :return: The tuple of the wrapped model and list of optimizers,
            in the same order they were passed in.
        """
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
