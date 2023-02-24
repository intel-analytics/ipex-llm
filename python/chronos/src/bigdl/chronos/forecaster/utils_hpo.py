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

from bigdl.nano.pytorch.lightning import LightningModule
from bigdl.nano.utils.common import invalidInputError
from typing import List
import pytorch_lightning as pl
from torchmetrics.metric import Metric
from torch.optim.lr_scheduler import _LRScheduler
import bigdl.nano.automl.hpo as hpo
import warnings
import torch
import inspect
from torch import nn, Tensor, fx
from torch.utils.data import TensorDataset, DataLoader
from bigdl.nano.automl.hpo.space import Space


@hpo.plmodel()
class GenericLightningModule(LightningModule):
    """A generic LightningMoudle for light-weight HPO."""

    def __init__(self,
                 model_creator,
                 optim_creator,
                 loss_creator,
                 data,
                 validation_data=None,
                 batch_size=32,
                 epochs=1,
                 metrics: List[Metric] = None,
                 scheduler: _LRScheduler = None,
                 num_processes=1,
                 model_config_keys=None,
                 data_config_keys=None,
                 optim_config_keys=None,
                 loss_config_keys=None,
                 **all_config):
        """
        Create LightningModule that exposes hyper parameters in init arguments.

        :param model_creator:   the model creator function.
        :param model_config:    model related configurations - argument of model_creator.
        :param data_config:     data related configurations - argument of model_creator.
        :param optim_creator:   the optimizer creator function.
        :param optim_config:    optim configurations - argument of optim_creator.
        :param loss_creator:    the loss creator function.
        :param loss_config:     the loss configurations - argument of loss_creator.
        :param data:            the train data
        :param validation_data: validation data required at present.
        :param scheduler:       learning rate scheduler.
        :param metrics:         list of metrics to calculate accuracy of the model.
        """
        self.data = data
        self.validation_data = validation_data
        # set batch size

        if batch_size % num_processes != 0:
            warnings.warn("'batch_size' cannot be divided with no remainder by "
                          "'num_processes'. We got 'batch_size' = {} and "
                          "'num_processes' = {}".
                          format(batch_size, num_processes))
        self.batch_size = max(1, batch_size // num_processes)
        self.epochs = epochs

        model_config = self._get_config_by_keys(model_config_keys, all_config)
        data_config = self._get_config_by_keys(data_config_keys, all_config)
        optim_config = self._get_config_by_keys(optim_config_keys, all_config)
        loss_config = self._get_config_by_keys(loss_config_keys, all_config)

        model = model_creator({**model_config, **data_config})
        loss = loss_creator(loss_config)
        optimizer = optim_creator(model, optim_config)

        invalidInputError(isinstance(model, nn.Module) and not
                          isinstance(model, pl.LightningModule),
                          "The created model must be instance of nn.Module but got {}"
                          .format(model.__class__))

        super().__init__(model=model, loss=loss, optimizer=optimizer,
                         scheduler=scheduler, metrics=metrics)

        _check_duplicate_metrics(self.metrics)
        self.save_hyperparameters(
            ignore=["model_creator", "optim_creator", "loss_creator",
                    "model_config_keys", "data_config_keys",
                    "optim_config_keys", "loss_config_keys",
                    "data", "validation_data", "metrics"])

    @staticmethod
    def _get_config_by_keys(keys, config):
        return {k: config[k] for k in keys}

    def validation_step(self, batch, batch_idx):
        """Define a single validation step."""
        y_hat = self(*batch)
        if self.loss:
            loss = self.loss(y_hat, batch[-1])  # use last output as target
            self.log("val/loss", loss, on_epoch=True,
                     prog_bar=True, logger=True)
        if self.metrics:
            acc = {_format_metric('val', metric, i): metric(y_hat, batch[-1])
                   for i, metric in enumerate(self.metrics)}
            self.log_dict(acc, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        """Define a single test step."""
        y_hat = self(*batch)
        if self.metrics:
            acc = {_format_metric('test', metric, i): metric(y_hat, batch[-1])
                   for i, metric in enumerate(self.metrics)}
            self.log_dict(acc, on_epoch=True, prog_bar=True, logger=True)

    def train_dataloader(self):
        """Create the train data loader."""
        return DataLoader(TensorDataset(torch.from_numpy(self.data[0]),
                                        torch.from_numpy(self.data[1])),
                          batch_size=self.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        """Create the validation data loader."""
        return DataLoader(TensorDataset(torch.from_numpy(self.validation_data[0]),
                                        torch.from_numpy(self.validation_data[1])),
                          batch_size=self.batch_size,
                          shuffle=True)


@hpo.plmodel()
class GenericTSTransformerLightningModule(pl.LightningModule):
    """A generic TS Transformer LightningMoudle for light-weight HPO."""

    def __init__(self,
                 model_creator,
                 loss_creator,
                 data,
                 validation_data=None,
                 batch_size=32,
                 epochs=1,
                 metrics: List[Metric] = None,
                 scheduler: _LRScheduler = None,
                 num_processes=1,
                 model_config_keys=None,
                 data_config_keys=None,
                 optim_config_keys=None,
                 loss_config_keys=None,
                 **all_config):
        """
        Create LightningModule that exposes hyper parameters in init arguments.

        :param model_creator:   the model creator function.
        :param model_config:    model related configurations - argument of model_creator.
        :param data_config:     data related configurations - argument of model_creator.
        :param optim_creator:   the optimizer creator function.
        :param optim_config:    optim configurations - argument of optim_creator.
        :param loss_creator:    the loss creator function.
        :param loss_config:     the loss configurations - argument of loss_creator.
        :param data:            the train data
        :param validation_data: validation data required at present.
        :param scheduler:       learning rate scheduler.
        :param metrics:         list of metrics to calculate accuracy of the model.
        """
        super().__init__()
        self.data = data
        self.validation_data = validation_data
        # set batch size

        if batch_size % num_processes != 0:
            warnings.warn("'batch_size' cannot be divided with no remainder by "
                          "'num_processes'. We got 'batch_size' = {} and "
                          "'num_processes' = {}".
                          format(batch_size, num_processes))
        self.batch_size = max(1, batch_size // num_processes)
        self.epochs = epochs

        model_config = self._get_config_by_keys(model_config_keys, all_config)
        data_config = self._get_config_by_keys(data_config_keys, all_config)
        optim_config = self._get_config_by_keys(optim_config_keys, all_config)
        loss_config = self._get_config_by_keys(loss_config_keys, all_config)
        pl.seed_everything(model_config["seed"], workers=True)
        self.model = model_creator({**model_config, **optim_config, **loss_config})
        self.loss = loss_creator(loss_config['loss'])

        invalidInputError(isinstance(self.model, pl.LightningModule),
                          "The created model must be instance of LightningModule but got {}"
                          .format(self.model.__class__))

        self.scheduler = scheduler
        self.metrics = metrics
        _check_duplicate_metrics(self.metrics)
        self.save_hyperparameters(
            ignore=["model_creator", "optim_creator", "loss_creator",
                    "model_config_keys", "data_config_keys",
                    "optim_config_keys", "loss_config_keys",
                    "data", "validation_data", "metrics", "model"])

    @staticmethod
    def _get_config_by_keys(keys, config):
        return {k: config[k] for k in keys}

    def forward(self, *args):
        """Same as torch.nn.Module.forward()."""
        nargs = len(inspect.getfullargspec(self.model.forward).args[1:])
        if isinstance(args, fx.Proxy):
            args = [args[i] for i in range(nargs)]
        else:
            args = args[:nargs]
        batch_x, batch_y, batch_x_mark, batch_y_mark = map(lambda x: x.float(), args)
        outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
        return outputs

    def training_step(self, batch, batch_idx):
        y_hat = self(*batch)
        target = batch[1][:, self.hparams.label_len:, :]
        loss = self.loss(y_hat, target)  # use last output as target
        self.log("train/loss", loss, on_epoch=True,
                 prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Define a single validation step."""
        y_hat = self(*batch)
        target = batch[1][:, self.hparams.label_len:, :]
        if self.loss:
            loss = self.loss(y_hat, target)  # use last output as target
            self.log("val/loss", loss, on_epoch=True,
                     prog_bar=True, logger=True)
        if self.metrics:
            acc = {_format_metric('val', metric, i): metric(y_hat, target)
                   for i, metric in enumerate(self.metrics)}
            self.log_dict(acc, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        """Define a single test step."""
        y_hat = self(*batch)
        target = batch[1][:, self.hparams.label_len:, :]
        if self.metrics:
            acc = {_format_metric('test', metric, i): metric(y_hat, target)
                   for i, metric in enumerate(self.metrics)}
            self.log_dict(acc, on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx):
        """Define a single test step."""
        y_hat = self(*batch)
        return y_hat

    def train_dataloader(self):
        """Create the train data loader."""
        return DataLoader(TensorDataset(torch.from_numpy(self.data[0]).float(),
                                        torch.from_numpy(self.data[1]).float(),
                                        torch.from_numpy(self.data[2]).float(),
                                        torch.from_numpy(self.data[3]).float()),
                          batch_size=self.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        """Create the validation data loader."""
        return DataLoader(TensorDataset(torch.from_numpy(self.validation_data[0]),
                                        torch.from_numpy(self.validation_data[1]),
                                        torch.from_numpy(self.validation_data[2]),
                                        torch.from_numpy(self.validation_data[3])),
                          batch_size=self.batch_size,
                          shuffle=False)

    def configure_optimizers(self):
        return self.model.configure_optimizers()


def _check_duplicate_metrics(metrics):
    metric_names = [metric.__name__ for i, metric in enumerate(metrics)]
    if len(metric_names) == len(set(metric_names)):
        return
    else:
        invalidInputError(False, "Duplicate metric names found.")


def _format_metric(prefix, metric, id=-1):
    """Format the metric as test/mean_squared_error, or val/mean_squared_error."""
    return "{}/{}".format(prefix, metric.__name__)


def _format_metric_str(prefix, metric):
    """Format the string metric."""
    if isinstance(metric, (list, tuple)):
        metrics = []
        for target_metric in metric:
            if target_metric == "latency":
                metrics.append(target_metric)
            else:
                metrics.append(_format_metric_str(prefix, target_metric))
        return metrics
    if isinstance(metric, str):
        from bigdl.chronos.metric.forecast_metrics import REGRESSION_MAP
        metric_func = REGRESSION_MAP.get(metric, None)
        invalidInputError(metric_func is not None,
                          "{} is not found in available metrics.".format(metric))
    return _format_metric(prefix, metric_func)


def _config_has_search_space(config):
        """Check if there's any search space in configuration."""
        for _, v in config.items():
            if isinstance(v, Space):
                return True
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, Space):
                        return True
        return False
