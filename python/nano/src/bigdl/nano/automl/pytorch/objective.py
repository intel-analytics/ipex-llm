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

from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl

from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from pytorch_lightning.utilities import rank_zero_deprecation
from pytorch_lightning.core.datamodule import LightningDataModule

from bigdl.nano.automl.hpo.backend import create_pl_pruning_callback
from bigdl.nano.utils.log4Error import invalidInputError
import inspect
import copy


def _is_creator(model):
    return inspect.ismethod(model) or inspect.isfunction(model)


class Objective(object):
    """The Tuning objective for HPO."""

    def __init__(self,
                 searcher,
                 model=None,
                 target_metric=None,
                 pruning=False,
                 **fit_kwargs):
        """
        Init the objective.

        :param: searcher: an HPOSearcher object which does the actual work.
        :param: model: a model instance or a creator function.
            Defaults to None.
        :param: target_metric: str(optional): target metric to optimize.
            Defaults to None.
        :param: pruning: bool (optional): whether to enable pruning.
            Defaults to False.
        throw: ValueError: _description_
        """
        self.searcher = searcher
        self.model_ = model
        self.target_metric = target_metric
        self.pruning = pruning
        self.fit_kwargs = fit_kwargs

        (self.train_dataloaders,
         self.val_dataloaders,
         self.datamodule) = self._fix_data_args(self.model, **fit_kwargs)

    @property
    def model(self):
        """Retrieve the model for tuning."""
        return self.model_

    def _pre_train(self, model, trial):
        # only do shallow copy and process/duplicate
        # specific args TODO: may need to handle more cases
        if self.pruning:
            callbacks = self.searcher.trainer.callbacks or []
            pruning_cb = create_pl_pruning_callback(trial, monitor=self.target_metric)
            callbacks.append(pruning_cb)
            self.searcher.trainer.callbacks = callbacks

        # links data to the trainer
        self.searcher.trainer.data_connector.attach_data(
            model,
            train_dataloaders=self.train_dataloaders,
            val_dataloaders=self.val_dataloaders,
            datamodule=self.datamodule
        )

    def _post_train(self, model):
        pass

    def _fix_data_args(self,
        model: "pl.LightningModule",
        train_dataloaders: Optional[Union[TRAIN_DATALOADERS, LightningDataModule]] = None,
        val_dataloaders: Optional[EVAL_DATALOADERS] = None,
        datamodule: Optional[LightningDataModule] = None,
        train_dataloader=None,  # noqa TODO: remove with 1.6
    ):
        if train_dataloader is not None:
            rank_zero_deprecation(
                "`trainer.search(train_dataloader)` is deprecated in v1.4"
                "and will be removed in v1.6."
                " Use `trainer.search(train_dataloaders)` instead. "
                "HINT: added 's'"
            )
            train_dataloaders = train_dataloader
        # if a datamodule comes in as the second arg, then fix it for the user
        if isinstance(train_dataloaders, LightningDataModule):
            datamodule = train_dataloaders
            train_dataloaders = None
        # If you supply a datamodule you can't supply train_dataloader or val_dataloaders
        if (train_dataloaders is not None or val_dataloaders is not None) and \
                datamodule is not None:
            invalidInputError(False,
                              "You cannot pass `train_dataloader`"
                              "or `val_dataloaders` to `trainer.search(datamodule=...)`")
        return train_dataloaders, val_dataloaders, datamodule

    def __call__(self, trial):
        """
        Execute Training and return target metric in each trial.

        :param: trial: a trial object which provides the hyperparameter combinition.
        :return: the target metric value.
        """
        # fit
        # hyperparameters = dict(n_layers=n_layers, dropout=dropout, output_dims=output_dims)
        # trainer.logger.log_hyperparams(hyperparameters)
        # trainer.fit(self.model, datamodule=datamodule)
        # return trainer.callback_metrics["val_acc"].item()

        if _is_creator(self.model):
            model = self.model(trial)
        else:
            # copy model so that the original model is not changed
            # Need tests to check this path
            model = copy.deepcopy(self.model)

        self._pre_train(model, trial)
        self.searcher._run(model)
        score = self.searcher.trainer.callback_metrics[self.target_metric].item()
        self._post_train(model)

        return score
