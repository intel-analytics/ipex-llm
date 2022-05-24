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

from pytorch_lightning.trainer.states import TrainerFn, TrainerStatus
from bigdl.nano.utils.log4Error import invalidInputError
from bigdl.nano.automl.hpo.backend import create_hpo_backend
from .objective import Objective
from ..hpo.search import (
    _search_summary,
    _end_search,
    _check_optimize_direction,
    _filter_tuner_args,
    _check_search_args,
)


class HPOSearcher:
    """Hyper Parameter Searcher. A Tuner-like class."""

    FIT_KEYS = {'train_dataloaders', 'val_dataloaders', 'datamodule', 'ckpt_path'}
    EXTRA_FIT_KEYS = {'max_epochs'}
    TUNE_CREATE_KEYS = {'storage', 'sampler', 'sampler_kwargs',
                        'pruner', 'pruner_kwargs', 'study_name', 'load_if_exists',
                        'direction', 'directions'}

    TUNE_RUN_KEYS = {'n_trials', 'timeout', 'n_jobs', 'catch', 'tune_callbacks',
                     'gc_after_trial', 'show_progress_bar'}

    def __init__(self, trainer: "pl.Trainer") -> None:
        """
        Init a HPO Searcher.

        :param trainer: The pl.Trainer object.
        """
        self.trainer = trainer
        self.objective = None
        self.model_class = pl.LightningModule
        self.study = None
        self.tune_end = False
        self._lazymodel = None
        self.backend = create_hpo_backend()

    def _run_search(self,
                    resume,
                    target_metric,
                    create_keys,
                    run_keys,
                    kwargs):

        # create study
        if self.study is None:
            if not resume:
                load_if_exists = False
                print("Starting a new tuning")
            else:
                load_if_exists = True
                print("Resume the last tuning...")

            study_create_kwargs = _filter_tuner_args(kwargs, create_keys)
            _check_optimize_direction(
                direction=study_create_kwargs.get('direction', None),
                directions=study_create_kwargs.get('directions', None),
                metric=target_metric)

            # prepare sampler and pruner args
            sampler_type = study_create_kwargs.get('sampler', None)
            if sampler_type:
                sampler_args = study_create_kwargs.get('sampler_kwargs', {})
                sampler = self.backend.create_sampler(sampler_type, sampler_args)
                study_create_kwargs['sampler'] = sampler
                study_create_kwargs.pop('sampler_kwargs', None)

            pruner_type = study_create_kwargs.get('pruner', None)
            if pruner_type:
                pruner_args = study_create_kwargs.get('pruner_kwargs', {})
                pruner = self.backend.create_pruner(pruner_type, pruner_args)
                study_create_kwargs['pruner'] = pruner
                study_create_kwargs.pop('pruner_kwargs', None)

            study_create_kwargs['load_if_exists'] = load_if_exists
            # create study
            self.study = self.backend.create_study(**study_create_kwargs)

        # renamed callbacks to tune_callbacks to avoid conflict with fit param
        study_optimize_kwargs = _filter_tuner_args(kwargs, run_keys)
        study_optimize_kwargs['callbacks'] = study_optimize_kwargs.get('tune_callbacks', None)
        study_optimize_kwargs.pop('tune_callbacks', None)
        study_optimize_kwargs['show_progress_bar'] = False
        # run optimize
        self.study.optimize(self.objective, **study_optimize_kwargs)

    def _pre_search(self,
                    model,
                    target_metric,
                    **kwargs):
        _check_search_args(
            search_args=self.search_kwargs,
            legal_keys=[
                HPOSearcher.FIT_KEYS,
                HPOSearcher.EXTRA_FIT_KEYS,
                HPOSearcher.TUNE_CREATE_KEYS,
                HPOSearcher.TUNE_RUN_KEYS
            ])

        isprune = True if kwargs.get('pruner', None) else False

        if self.objective is None:
            # target_metric = self._fix_target_metric(target_metric, search_kwargs)
            fit_kwargs = _filter_tuner_args(kwargs, HPOSearcher.FIT_KEYS)
            self.objective = Objective(
                searcher=self,
                model=model._model_build,
                target_metric=target_metric,
                pruning=isprune,
                **fit_kwargs,
            )

        # TODO do we need to set these trainer states?
        self.trainer.state.fn = TrainerFn.TUNING
        self.trainer.state.status = TrainerStatus.RUNNING
        self.trainer.tuning = True

    def _post_search(self):
        self.tune_end = False
        self.trainer.tuning = False
        # Force the train dataloader to reset as the batch size has changed
        # self.trainer.reset_train_dataloader(model)
        # self.trainer.reset_val_dataloader(model)
        self.trainer.state.status = TrainerStatus.FINISHED
        invalidInputError(self.trainer.state.stopped,
                          "trainer state should be stopped")

    def search(self,
               model,
               resume: bool = False,
               target_metric=None,
               **kwargs):
        """
        Run HPO Searcher. It will be called in Trainer.search().

        :param model: The model to be searched. It should be an automodel.
        :param resume: whether to resume the previous or start a new one,
            defaults to False.
        :param target_metric: the object metric to optimize,
            defaults to None.
        :param return: the model with study meta info attached.
        """
        self.search_kwargs = kwargs or {}
        self.target_metric = target_metric

        self._pre_search(model, target_metric, **kwargs)
        self._run_search(resume=resume,
                         target_metric=self.target_metric,
                         create_keys=HPOSearcher.TUNE_CREATE_KEYS,
                         run_keys=HPOSearcher.TUNE_RUN_KEYS,
                         kwargs=self.search_kwargs)

        self._post_search()
        # it is not possible to know the best trial before runing search,
        # so just apply the best trial at end of each search
        self._lazymodel = _end_search(study=self.study,
                                      model_builder=model._model_build,
                                      use_trial_id=-1)
        return self._lazymodel

    def search_summary(self):
        """
        Retrive a summary of trials.

        :return: A summary of all the trials. Currently the entire study is
            returned to allow more flexibility for further analysis and visualization.
        """
        return _search_summary(self.study)

    def end_search(self, use_trial_id=-1):
        """
        Put an end to tuning.

        Use the specified trial or best trial to init and build the model.

        :param use_trial_id: int(optional) params of which trial to be used.
            Defaults to -1.
        :throw: ValueError: error when tune is not called before end_search.
        """
        self._lazymodel = _end_search(study=self.study,
                                      model_builder=self._model_build,
                                      use_trial_id=use_trial_id)
        # TODO Next step: support retrive saved model instead of retrain from hparams
        self.tune_end = True
        return self._lazymodel

    def _run(self, *args: Any, **kwargs: Any) -> None:
        """`_run` wrapper to set the proper state during tuning,\
        as this can be called multiple times."""
        # last `_run` call might have set it to `FINISHED`
        self.trainer.state.status = TrainerStatus.RUNNING
        self.trainer.training = True
        self.trainer._run(*args, **kwargs)
        self.trainer.tuning = True
