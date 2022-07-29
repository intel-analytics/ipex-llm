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
from collections import OrderedDict
import inspect
from typing import List

import torch
from torchmetrics.metric import Metric
import pytorch_lightning as pl
from torch import nn, Tensor, fx
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from bigdl.nano.utils.log4Error import invalidInputError


class LightningModule(pl.LightningModule):
    """
    A wrapper LightningMoudle for common PyTorch models.

    This class implements common methods in LightningModule, so that classic pytorch
    supervised learning model only needs to supply module, loss and optimizer to create
    a LightningModule.
    """

    def __init__(self, model: nn.Module,
                 loss: _Loss = None,
                 optimizer: Optimizer = None,
                 scheduler: _LRScheduler = None,
                 metrics: List[Metric] = None,
                 channel_last = False,
                 ipex=False,
                 dtype=None,
                 jit=None,
                 example_input_array=None):
        """
        Create a LightningMoudle that integrates pytorch modules, loss, optimizer.

        :param model:       Pytorch model to be converted.
        :param loss:        A torch loss function.
        :param optimizer:   A torch optimizer.
        :param scheduler:   A torch scheduler.
        :param metrics:     A list of metrics to calculate accuracy of the model.
        :param jit:     Specify JIT mode, defaults to None which means JIT disabled. Other options
                        can be `trace` or `script`
        :param example_input_array:     A tuple of example inputs for forward, trace, script, etc.
        """
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics
        self.example_input_array = example_input_array
        self._status = {
            "ModelType": type(self).__name__,
            "use_ipex": False,
            "channel_last": False,
            "use_jit": False,
            "dtype": "FP32"
        }
        self.channel_last(channel_last)
        if ipex:
            self.ipex(dtype)
        self._jit = None
        self.jit(jit, example_input_array)
        self._forward_args_ = None

    @property
    def forward_args(self):  # noqa
        if self._forward_args_ is None:
            self._forward_args_ = inspect.getfullargspec(self.model.forward).args[1:]
        return self._forward_args_

    @forward_args.setter
    def forward_args(self, forward_args):
        self._forward_args_ = forward_args

    @property
    def _nargs(self):
        return len(self.forward_args)

    def compile(self,
                loss: _Loss = None, optimizer: Optimizer = None,
                scheduler: _LRScheduler = None,
                metrics: List[Metric] = None):
        """
        Compile a LightningMoudle withloss, optimizer, metrics, schedulers.

        :param loss:        A torch loss function.
        :param optimizer:   A torch optimizer.
        :param scheduler:   A torch scheduler.
        :param metrics:     A list of metrics to calculate accuracy of the model.
        """
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if scheduler is not None:
            self.scheduler = scheduler
        if metrics is not None:
            self.metrics = metrics

    def on_forward_start(self, inputs):
        return inputs

    def forward_step(self, *args):
        if isinstance(args, fx.Proxy):
            args = [args[i] for i in range(self._nargs)]
        else:
            args = args[:self._nargs]
        return self.model(*args)

    def on_forward_end(self, outputs):
        return outputs

    def forward(self, *args):
        """Same as torch.nn.Module.forward()."""
        inputs = self.on_forward_start(args)
        outputs = self.forward_step(*inputs)
        return self.on_forward_end(outputs)

    def on_train_start(self) -> None:
        """Called at the beginning of training after sanity check."""
        invalidInputError(self.loss, "Loss must not be None for training.")

    def training_step(self, batch, batch_idx):
        """Define a single training step, return a loss tensor."""
        y_hat = self(*batch[:self._nargs])
        loss = self.loss(y_hat, batch[-1])  # use last output as target
        self.log("train/loss", loss, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Define a single validation step."""
        y_hat = self(*batch[:self._nargs])
        if self.loss:
            loss = self.loss(y_hat, batch[-1])  # use last output as target
            self.log("val/loss", loss, on_epoch=True,
                     prog_bar=True, logger=True)
        if self.metrics:
            acc = {"val/{}_{}".format(type(metric).__name__, i): metric(y_hat, batch[-1])
                   for i, metric in enumerate(self.metrics)}
            self.log_dict(acc, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        """Define a single test step."""
        y_hat = self(*batch[:self._nargs])
        if self.metrics:
            acc = {"test/{}_{}".format(type(metric).__name__, i): metric(y_hat, batch[-1])
                   for i, metric in enumerate(self.metrics)}
            self.log_dict(acc, on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Define a single predict step."""
        return self(*batch[:self._nargs])

    def configure_optimizers(self):
        """Setup the optimizers for this module, and return optimizers and schedulers."""
        optimizers = [self.optimizer]
        schedulers = []
        if self.scheduler:
            schedulers.append(self.scheduler)
        return optimizers, schedulers

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        """Same as LightningModule.load_state_dict, execept falling back to pytorch."""
        try:
            super().load_state_dict(state_dict)
        except RuntimeError:
            self.model.load_state_dict(state_dict)

    def jit(self, mode='trace', input_sample=None):
        if mode is None:
            return
        if input_sample is None:
            input_sample = self.example_input_array
        else:
            self.example_input_array = input_sample
        if mode == 'trace':
            self.model = torch.jit.trace(self.model, input_sample)
            self.model = torch.jit.freeze(self.model)
        else:
            invalidInputError(
                not mode == 'script',
                errMsg="JIT supports 'trace' and 'script' only, {} is not valid".format(mode)
            )
            self.model = torch.jit.script(self.model, input_sample)
        self._jit = mode

    def ipex(self, dtype):
        import intel_extension_for_pytorch as ipex
        self._status['use_ipex'] = True
        self._status['precision'] = str(dtype)
        self.model = ipex.optimize(self.model, dtype=dtype)

    def channel_last(self, mode=True):
        self._status['channel_last'] = mode
        if mode:
            self.model = self.model.to(memory_format=torch.channels_last)
        else:
            self.model = self.model.to(memory_format=torch.preserve_format)

    @property
    def status(self):
        return self._status
