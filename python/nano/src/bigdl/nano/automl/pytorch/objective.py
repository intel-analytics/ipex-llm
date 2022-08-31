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

from collections import namedtuple
from typing import Optional, Tuple, Union

import pytorch_lightning as pl

from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from pytorch_lightning.utilities import rank_zero_deprecation
from pytorch_lightning.core.datamodule import LightningDataModule

from bigdl.nano.pytorch.trainer import Trainer
from bigdl.nano.automl.hpo.backend import create_pl_pruning_callback
from bigdl.nano.utils.log4Error import invalidInputError
from ._helper import LatencyCallback, _remove_metric_prefix
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
                 acceleration=False,
                 input_sample=None,
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
        :param acceleration: Whether to automatically consider the model after
            inference acceleration in the search process. It will only take
            effect if target_metric contains "latency". Default value is False.
        throw: ValueError: _description_
        """
        self.searcher = searcher
        self.model_ = model
        self.target_metric = target_metric
        self.mo_hpo = isinstance(target_metric, (list, tuple)) \
            and len(self.target_metric) > 1
        # add automatic support for latency
        if self.mo_hpo and "latency" in self.target_metric:
            callbacks = self.searcher.trainer.callbacks or []
            callbacks.append(LatencyCallback())
            self.searcher.trainer.callbacks = callbacks
            self.acceleration = acceleration
            self.input_sample = input_sample
        else:
            self.acceleration = False

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
        self.searcher.trainer._data_connector.attach_data(  # type: ignore
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

    def _auto_acceleration(self, model, scores):
        # for compatibility with metric of Chronos, remove the prefix before '/'
        format_target_metric = _remove_metric_prefix(self.target_metric)
        Score = namedtuple("Score", format_target_metric)
        best_score = Score(*scores)
        optim_model_type = "original"
        model.eval()

        # here we suppose nn.model is attached to plmodel by attribute "model"
        original_model = model.model
        # may define a default_range later
        for optimization in ['openvino', 'onnxruntime', 'jit']:
            # may enable more optimizations later
            try:
                if optimization == 'jit':
                    optim_model = Trainer.trace(original_model, input_sample=self.input_sample,
                                                accelerator=optimization,
                                                use_ipex=True, channels_last=False)
                else:
                    optim_model = Trainer.trace(original_model, input_sample=self.input_sample,
                                                accelerator=optimization)
            except Exception:
                # some optimizations may fail, just skip and try next
                continue
            model.model = optim_model
            model.forward_args = optim_model.forward_args
            self.searcher._validate(model, self.val_dataloaders)
            optim_score = []
            for metric in self.target_metric:
                score = self.searcher.trainer.callback_metrics[metric].item()
                optim_score.append(score)
            optim_score = Score(*optim_score)
            # Here, we use usable to represent whether the optimized model can get smaller latency
            # with other performance indicators basically unchanged
            usable = True
            for metric in format_target_metric:
                if metric != "latency":
                    if abs(getattr(optim_score, metric) - getattr(best_score, metric)) >= \
                            0.01 * getattr(best_score, metric):
                        usable = False
                        break
                else:
                    if optim_score.latency > best_score.latency:
                        usable = False
                        break
            if usable:
                best_score = optim_score
                optim_model_type = optimization
        scores = tuple(best_score)
        return scores, optim_model_type

    def __call__(self, trial):
        """
        Execute Training and return target metric in each trial.

        :param: trial: a trial object which provides the hyperparameter combinition.
        :return: the target metric value.
        """
        if _is_creator(self.model):
            model = self.model(trial)
        else:
            # copy model so that the original model is not changed
            # Need tests to check this path
            model = copy.deepcopy(self.model)

        self._pre_train(model, trial)
        self.searcher._run(model)
        if self.mo_hpo:
            scores = []
            for metric in self.target_metric:
                score = self.searcher.trainer.callback_metrics[metric].item()
                scores.append(score)
        else:
            scores = self.searcher.trainer.callback_metrics[self.target_metric].item()
        if self.acceleration:
            scores, optimization = self._auto_acceleration(model, scores)
            # via user_attr returns the choosed optimization corresponding
            # to the minimum latency
            trial.set_user_attr("optimization", optimization)
        self._post_train(model)
        return scores
