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

from typing import Any, Optional
from logging import warning

from torch import nn
from torch import Tensor
from torch.optim import Optimizer
import pytorch_lightning.lite as lite
from pytorch_lightning.lite.wrappers import _LiteModule

from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10
from bigdl.nano.pytorch.strategies import create_IPEXStrategy


class LightningLite(lite.LightningLite):
    """
    LightningLite for BigDL-Nano pytorch.

    This LightningLite extends PyTorch Lightning's LightningLite by adding
    various options to accelerate pytorch training.
    """

    def __init__(self, num_processes: int = 1,
                 use_ipex: bool = False,
                 enable_bf16: bool = False,
                 distributed_backend: str = "subprocess",
                 *args, **kwargs) -> None:
        """
        Create a LightningLite with nano acceleration.

        :param num_processes: number of processes in distributed training, defaults to 1
        :param use_ipex: whether use ipex acceleration, defaults to False
        :param enable_bf16: whether use bf16 acceleration, defaults to False
        :param distributed_backend: use which backend in distributed mode, defaults to "subprocess"
        """
        # Check keyword arguments
        if "strategy" in kwargs:
            warning(f"""strategy will be specified by bigdl-nano,
            strategy entered {kwargs['strategy']} will be ignored.""")

        self.use_ipex = use_ipex
        self.enable_bf16 = enable_bf16

        strategy = None
        if num_processes == 1:
            strategy = create_IPEXStrategy(enable_bf16=self.enable_bf16)
        else:
            pass

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
        # add IPEX's optimization
        known_strategies = ["ipex"]
        if self._strategy is not None and self._strategy.strategy_name in known_strategies:
            model, optimizers = self._strategy._setup_lite(model, *optimizers)

        return super().setup(model, *optimizers, move_to_device=move_to_device)

    def backward(self,
                 tensor: Tensor,
                 *args: Any,
                 model: Optional[_LiteModule] = None,
                 **kwargs: Any) -> None:
        """IPEX 1.9 requires moving loss tensor to xpu, we do it here."""
        if self.use_ipex and TORCH_VERSION_LESS_1_10:
            tensor = tensor.to(self.device)
        super().backward(tensor, *args, model=model, **kwargs)
