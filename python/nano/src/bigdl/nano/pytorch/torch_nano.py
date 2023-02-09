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


from typing import Any, Union, List, Optional
from logging import warning
from functools import partial, wraps
from abc import abstractmethod

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.parallel.distributed import DistributedDataParallel
from pytorch_lightning.lite import LightningLite
from pytorch_lightning.lite.wrappers import _LiteModule, _LiteOptimizer
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.strategies import Strategy
from pytorch_lightning.strategies import DeepSpeedStrategy

from bigdl.nano.utils.common import _avx512_checker
from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_11
from bigdl.nano.deps.ipex.ipex_api import ipex_optimize
from bigdl.nano.pytorch.strategies import IPEXStrategy, DDPSpawnStrategy, \
    DDPSubprocessStrategy, create_ray_strategy, DDPK8sStrategy


class _TorchNanoModule(_LiteModule):
    def __init__(self, module, precision_plugin, channels_last) -> None:
        super().__init__(module, precision_plugin)
        self.channels_last = channels_last

    def state_dict(self, *args, **kwargs):
        if isinstance(self.module, DistributedDataParallel):
            return self.module.module.state_dict(*args, **kwargs)
        else:
            return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        invalidInputError(False, "TorchNano doesn't support loading state dict, "
                          "please load it using original pytorch model")

    def __getattr__(self, name: str):
        # automatically unwrap attributes access of _LiteModule,
        # always throw a single-level exception when the attribute doesn't exist
        # for a more user-friendly exception message
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass

        # When using multi-instance training, self.module will be DistributedDataParallel(DDP),
        # otherwise, `self.module` will be original module.
        if isinstance(self.module, DistributedDataParallel):
            # just in case that users try to access an attribute of DDP
            # or an attribute of both DDP and original model,
            # we should first try to find it in DDP
            try:
                return getattr(self.module, name)
            except AttributeError:
                pass
            return getattr(self.module.module, name)
        else:
            return getattr(self.module, name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Casts all inputs to the right memory format."""
        if self.channels_last:
            def _convert_to_channels_last(t: torch.Tensor) -> torch.Tensor:
                if t.dim() == 4:
                    return t.to(memory_format=torch.channels_last)  # type: ignore
                return t
            args, kwargs = apply_to_collection([args, kwargs], function=_convert_to_channels_last,
                                               dtype=torch.Tensor)
        return super().forward(*args, **kwargs)


class _TorchNanoOptimizer(_LiteOptimizer):
    def __init__(self, optimizer: Optimizer, strategy: Strategy,
                 auto_lr: bool, num_processes: Optional[int]) -> None:
        super().__init__(optimizer, strategy)
        self.cur_lr_ratio = 1.0
        self.max_lr_ratio = num_processes
        self.cur_step = 0
        self.max_step = 1000
        self.auto_lr = auto_lr

    def step(self, closure=None) -> Any:     # type: ignore
        if not self.auto_lr or self.max_lr_ratio is None or self.max_lr_ratio == 1:
            return super().step(closure)
        else:
            # adjust learning rate
            base_lrs = []
            for param_group in self.optimizer.param_groups:
                base_lr = param_group['lr']
                base_lrs.append(base_lr)
                param_group['lr'] = base_lr * self.cur_lr_ratio

            # call step
            ret = super().step(closure=closure)

            # restore learning rate
            for param_group, base_lr in zip(self.optimizer.param_groups, base_lrs):
                param_group['lr'] = base_lr

            if self.cur_step < self.max_step:
                self.cur_step += 1
                self.cur_lr_ratio = (self.max_lr_ratio - 1) * self.cur_step / self.max_step + 1

            return ret


distributed_backends = ["spawn", "ray", "subprocess", "k8s"]

backends_class_map = {
    "spawn": DDPSpawnStrategy,
    "subprocess": DDPSubprocessStrategy,
    "ray": create_ray_strategy,
    "k8s": DDPK8sStrategy
}


class TorchNano(LightningLite):
    """
    TorchNano for BigDL-Nano pytorch.

    It can be used to accelerate custom pytorch training loops with very few code changes.
    """

    def __init__(self, num_processes: Optional[int] = None,
                 use_ipex: bool = False,
                 distributed_backend: str = "subprocess",
                 precision: Union[str, int] = 32,
                 cpu_for_each_process: Optional[List[List[int]]] = None,
                 channels_last: bool = False,
                 auto_lr: bool = True,
                 *args, **kwargs) -> None:
        """
        Create a TorchNano with nano acceleration.

        :param num_processes: number of processes in distributed training, defaults to ``1``
        :param use_ipex: whether use ipex acceleration, defaults to ``False``
        :param distributed_backend: use which backend in distributed mode, defaults to
            ``'subprocess'``, now avaiable backends are ``'spawn'``, ``'subprocess'`` and ``'ray'``
        :param precision: Double precision (``64``), full precision (``32``),
            half precision (``16``) or bfloat16 precision (``'bf16'``), defaults to ``32``.
            Enable ipex bfloat16 weight prepack when ``use_ipex=True`` and ``precision='bf16'``
        :param cpu_for_each_process: specify the cpu cores which will be used by each process,
            if ``None``, cpu cores will be distributed evenly by all processes,
            only take effect when ``num_processes`` > 1
        :param channels_last: whether convert input to channels last memory formats,
            defaults to ``False``.
        :param auto_lr: whether to scale the learning rate linearly by ``num_processes`` times.
            Defaults to ``True``.
            If ``num_processes=1`` or other ``lr_scheduler`` is set, ``auto_lr`` will be ignored.
        """
        self.num_processes = num_processes
        self.use_ipex = use_ipex
        self.dtype = None
        self.cpu_for_each_process = cpu_for_each_process
        self.channels_last = channels_last
        self.auto_lr = auto_lr

        if self.use_ipex and precision == 'bf16':
            # Enable ipex bfloat16 weight prepack and disable native AMP
            self.dtype = torch.bfloat16
            precision = 32

        # Confirm if cpu supports AVX512
        if self.use_ipex and not _avx512_checker():
            if TORCH_VERSION_LESS_1_11:
                warning("Enable ipex<=1.10 in a cpu instruction set"
                        " without avx512 will crash."
                        "Fall back to regular pytorch.")
                self.use_ipex = False
            elif self.dtype == torch.bfloat16:
                warning("Enable IPEX bfloat16 in a cpu instruction set"
                        " without avx512 will crash. "
                        "Using 32-bit precision")
                self.dtype = None

        kwargs['precision'] = precision

        if self.num_processes is None and distributed_backend != "k8s":
            self.num_processes = 1

        if self.num_processes == 1:
            if self.use_ipex:
                strategy = IPEXStrategy(dtype=self.dtype)
            else:
                strategy = None     # type: ignore
        elif distributed_backend in backends_class_map:
            cls = backends_class_map[distributed_backend]
            strategy = cls(num_processes=self.num_processes,   # type: ignore
                           cpu_for_each_process=self.cpu_for_each_process,
                           use_ipex=self.use_ipex,
                           dtype=self.dtype)
        else:
            warning(f"BigDL-Nano doesn't support '{distributed_backend}' backend now, "
                    f"'{distributed_backend}' strategy of pytorch_lightning will be used. "
                    f"Supported backends are 'spawn', 'subprocess' and 'ray'.")
            strategy = distributed_backend  # type: ignore

        kwargs["strategy"] = strategy
        super().__init__(*args, **kwargs)

        setattr(self, "train", partial(self._run_impl, self.train))

    def _setup(
        self,
        model: nn.Module,
        optimizers: List[Optimizer],
        move_to_device: bool = True,
    ) -> Any:
        """Used to replace LightningLite's setup method."""
        if self.channels_last:
            model = model.to(memory_format=torch.channels_last)  # type: ignore
        # LightningLite won't call `Strategy.setup()` method,
        # in which we add IPEX's optimization when using `trainer`.

        # When we call `TorchNano().train()`, it will call
        # `Strategy.setup_environment()` -> `Lanucher.launch()` -> user defined `train()` method.

        # However the model and optimizers haven't been specified when calling these three methods,
        # so we have to add optimizations in this method, which will be called in
        # user defined `train()` method.

        self._validate_setup(model, optimizers)

        if move_to_device:
            model = self._move_model_to_device(model=model, optimizers=optimizers)

        model, optimizers = self._strategy._setup_model_and_optimizers(model, optimizers)

        # IPEX bfloat16 optimization will cast model parameters to `torch.bfloat16`
        # which is not supported by ddp currently,
        # so add IPEX 1.11's optimization after `_setup_model`
        if self.use_ipex:
            ret = ipex_optimize(model, optimizers=optimizers, inplace=False, dtype=self.dtype)
            if isinstance(ret, tuple):
                model, optimizers = ret[0], [ret[1]]
            else:
                model = ret

        model = _TorchNanoModule(model, self._precision_plugin, self.channels_last)
        optimizers = [_TorchNanoOptimizer(optimizer, self._strategy,    # type: ignore
                                          self.auto_lr, self.num_processes)
                      for optimizer in optimizers]
        self._models_setup += 1
        return model, optimizers

    def setup(self, model: nn.Module,    # type: ignore[override]
              optimizer: Union[Optimizer, List[Optimizer]],
              *dataloaders: DataLoader, move_to_device: bool = True):
        """
        Setup model, optimizers and dataloaders for accelerated training.

        :param model: A model to setup
        :param optimizer: The optimizer(s) to setup
        :param *dataloaders: The dataloader(s) to setup
        :param move_to_device: If set ``True`` (default), moves the model to the correct device.
            Set this to ``False`` and alternatively use :meth:`to_device` manually.
        :return: The tuple of the wrapped model, optimizer, loss_func and dataloaders,
            in the same order they were passed in.
        """
        # convert single optimizer to a optimizer list
        optimizers = [optimizer] if isinstance(optimizer, Optimizer) else optimizer

        model, optimizers = self._setup(model, optimizers, move_to_device=move_to_device)
        dataloaders = self.setup_dataloaders(*dataloaders,  # type: ignore
                                             move_to_device=move_to_device)
        # convert optimizer list to single optimizer
        optimizer = optimizers[0] if isinstance(optimizer, Optimizer) else optimizers
        if len(dataloaders) == 0:
            return model, optimizer
        else:
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


def _search_setup_args(_models, _optimizers, _dataloaders, args):
    for idx, value in enumerate(args):
        if isinstance(value, DataLoader):
            _dataloaders.append((value, args, idx))

        if isinstance(value, nn.Module) and not isinstance(value, torch.nn.modules.loss._Loss):
            _models.append((value, args, idx))

        if isinstance(value, Optimizer):
            _optimizers.append((value, args, idx))


def _update_args(objs, obj_pos):

    # obj_pos is a lists of (object, args|kwargs, idx)

    for obj, pos in zip(objs, obj_pos):
        _, arg, idx = pos
        arg[idx] = obj


class _DecoratedTorchNano(TorchNano):

    def train(self, func, *inner_args, **inner_kwargs):

        # todo: need to be able to let user specify which arg is model|optimizer, if
        # the search does not work.

        # search for model, optimizer and dataloaders in the param list
        # save the result in lists of (object, args|kwargs, idx)
        _model_pos = []
        _optimizer_pos = []
        _data_loader_pos = []

        _inner_args = list(inner_args)
        _search_setup_args(_model_pos, _optimizer_pos, _data_loader_pos, _inner_args)
        _search_setup_args(_model_pos, _optimizer_pos, _data_loader_pos, inner_kwargs)

        invalidInputError(len(_model_pos) == 1,
                          "there should be only one nn.Module "
                          f"in the function parameter list, but got {len(_model_pos)}")

        # get the objec to be setup
        _model = _model_pos[0][0]
        _optimizers = [opt[0] for opt in _optimizer_pos]
        _dataloaders = [opt[0] for opt in _data_loader_pos]

        # call setup, the purpose of the decorator
        _setup_model, _setup_optimizers = self.setup(_model, _optimizers)
        _setup_dataloaders = self.setup_dataloaders(*_dataloaders)
        if len(_dataloaders) == 1:
            _setup_dataloaders = [_setup_dataloaders]

        # update the function param list
        _update_args([_setup_model], _model_pos)
        _update_args(_setup_optimizers, _optimizer_pos)
        _update_args(_setup_dataloaders, _data_loader_pos)

        return func(*_inner_args, **inner_kwargs)


def nano(num_processes: Optional[int] = None,
         use_ipex: bool = False,
         distributed_backend: str = "subprocess",
         precision: Union[str, int] = 32,
         cpu_for_each_process: Optional[List[List[int]]] = None,
         channels_last: bool = False,
         auto_lr: bool = True,
         *args, **kwargs):
    """
    Run ``TorchNano.train`` through a convenient decorator function.

    :param num_processes: number of processes in distributed training, defaults to ``1``
    :param use_ipex: whether use ipex acceleration, defaults to ``False``
    :param distributed_backend: use which backend in distributed mode, defaults to
        ``'subprocess'``, now avaiable backends are ``'subprocess'`` and ``'ray'``.
        ``bigdl.nano.pytorch.nano`` decorator does not support ``'spawn'``.
    :param precision: Double precision (``64``), full precision (``32``), half precision (``16``)
        or bfloat16 precision (``'bf16'``), defaults to ``32``.
        Enable ipex bfloat16 weight prepack when ``use_ipex=True`` and ``precision='bf16'``
    :param cpu_for_each_process: specify the cpu cores which will be used by each process,
        if ``None``, cpu cores will be distributed evenly by all processes,
        only take effect when ``num_processes`` > 1
    :param channels_last: whether convert input to channels last memory formats,
        defaults to ``False``.
    :param auto_lr: whether to scale the learning rate linearly by ``num_processes`` times.
        Defaults to ``True``.
        If ``num_processes=1`` or other ``lr_scheduler`` is set, ``auto_lr`` will be ignored.
    """
    if "strategy" in kwargs:
        strategy = kwargs["strategy"]
        if strategy == "deepspeed" or isinstance(strategy, DeepSpeedStrategy):
            invalidInputError(False, "bigdl.nano.pytorch.nano do not support deepspeed strategy")

    # spawn has a wierd pickle error
    invalidInputError(distributed_backend != "spawn",
                      "bigdl.nano.pytorch.nano do not support spawn")

    def decorator(func):

        # todo check the func signature
        @wraps(func)
        def wrapper(*inner_args, **inner_kwargs):

            return _DecoratedTorchNano(num_processes=num_processes,
                                       use_ipex=use_ipex,
                                       distributed_backend=distributed_backend,
                                       precision=precision,
                                       cpu_for_each_process=cpu_for_each_process,
                                       channels_last=channels_last,
                                       auto_lr=auto_lr,
                                       *args, **kwargs).train(func, *inner_args, **inner_kwargs)

        return wrapper
    return decorator
