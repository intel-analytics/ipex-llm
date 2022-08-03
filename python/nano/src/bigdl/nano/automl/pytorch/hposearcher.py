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
import pytorch_lightning as pl
import copy
import math
from pytorch_lightning.trainer.states import TrainerFn, TrainerStatus
from bigdl.nano.utils.log4Error import invalidInputError
from bigdl.nano.automl.hpo.backend import create_hpo_backend, SamplerType
from .objective import Objective
from ._helper import ResetCallback, CustomEvaluationLoop
from bigdl.nano.automl.utils.parallel import run_parallel
from bigdl.nano.automl.hpo.search import (
    _search_summary,
    _end_search,
    _create_study,
    _validate_args,
    _prepare_args,
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

    def __init__(self, trainer: "pl.Trainer", num_processes: int = 1) -> None:
        """
        Init a HPO Searcher.

        :param trainer: The pl.Trainer object.
        """
        self.trainer = trainer
        self.num_process = num_processes
        if num_processes == 1:
            callbacks = self.trainer.callbacks or []
            callbacks.append(ResetCallback())
            self.trainer.callbacks = callbacks

        self.model_class = pl.LightningModule
        self.study = None
        self.objective = None
        self.tune_end = False
        self._lazymodel = None
        self.backend = create_hpo_backend()
        self.create_kwargs = None
        self.run_kwargs = None
        self.fit_kwargs = None

    def _create_objective(self, model, target_metric, create_kwargs, acceleration,
                          input_sample, fit_kwargs):
        # target_metric = self._fix_target_metric(target_metric, search_kwargs)
        isprune = True if create_kwargs.get('pruner', None) else False
        self.objective = Objective(
            searcher=self,
            model=model._model_build,
            target_metric=target_metric,
            pruning=isprune,
            acceleration=acceleration,
            input_sample=input_sample,
            **fit_kwargs,
        )

    def _run_search(self):
        # TODO do we need to set these trainer states?
        self.trainer.state.fn = TrainerFn.TUNING
        self.trainer.state.status = TrainerStatus.RUNNING
        self.trainer.tuning = True

        # run optimize
        self.study.optimize(self.objective, **self.run_kwargs)

        self.tune_end = False
        self.trainer.tuning = False
        # Force the train dataloader to reset as the batch size has changed
        # self.trainer.reset_train_dataloader(model)
        # self.trainer.reset_val_dataloader(model)
        self.trainer.state.status = TrainerStatus.FINISHED
        invalidInputError(self.trainer.state.stopped,
                          "trainer state should be stopped")

    def _run_search_n_procs(self, n_procs=4):
        new_searcher = copy.deepcopy(self)
        n_trials = new_searcher.run_kwargs.get('n_trials', None)
        if n_trials:
            subp_n_trials = math.ceil(n_trials / n_procs)
            new_searcher.run_kwargs['n_trials'] = subp_n_trials
        run_parallel(func=new_searcher._run_search,
                     kwargs={},
                     n_procs=n_procs)

    def search(self,
               model,
               resume=False,
               target_metric=None,
               n_parallels=1,
               acceleration=False,
               input_sample=None,
               **kwargs):
        """
        Run HPO Searcher. It will be called in Trainer.search().

        :param model: The model to be searched. It should be an automodel.
        :param resume: whether to resume the previous or start a new one,
            defaults to False.
        :param target_metric: the object metric to optimize,
            defaults to None.
        :param acceleration: Whether to automatically consider the model after
            inference acceleration in the search process. It will only take
            effect if target_metric contains "latency". Default value is False.
        :param input_sample: A set of inputs for trace, defaults to None if you have
            trace before or model is a LightningModule with any dataloader attached.
        :param return: the model with study meta info attached.
        """
        search_kwargs = kwargs or {}
        self.target_metric = target_metric

        _validate_args(search_kwargs,
                       self.target_metric,
                       legal_keys=[HPOSearcher.FIT_KEYS,
                                   HPOSearcher.EXTRA_FIT_KEYS,
                                   HPOSearcher.TUNE_CREATE_KEYS,
                                   HPOSearcher.TUNE_RUN_KEYS])

        _sampler_kwargs = model._lazyobj.sampler_kwargs
        user_sampler_kwargs = kwargs.get("sampler_kwargs", {})
        _sampler_kwargs.update(user_sampler_kwargs)
        if "sampler" in kwargs and kwargs["sampler"] in [SamplerType.Grid]:
            search_kwargs["sampler_kwargs"] = _sampler_kwargs

        (self.create_kwargs, self.run_kwargs, self.fit_kwargs) \
            = _prepare_args(search_kwargs,
                            HPOSearcher.TUNE_CREATE_KEYS,
                            HPOSearcher.TUNE_RUN_KEYS,
                            HPOSearcher.FIT_KEYS,
                            self.backend)

        # create study
        if self.study is None:
            self.study = _create_study(resume, self.create_kwargs, self.backend)

        if self.objective is None:
            self._create_objective(model, self.target_metric, self.create_kwargs, acceleration,
                                   input_sample, self.fit_kwargs)

        if n_parallels and n_parallels > 1:
            invalidInputError(self.create_kwargs.get('storage', "").strip() != "",
                              "parallel search is not supported"
                              " when in-mem storage is used (n_parallels must be 1)")

            self._run_search_n_procs(n_procs=n_parallels)
        else:
            self._run_search()

        # it is not possible to know the best trial before runing search,
        # so just apply the best trial at end of each search.
        # a single best trial cannot be retrieved from a multi-objective study.
        if not self.objective.mo_hpo:
            self._lazymodel = _end_search(study=self.study,
                                          model_builder=model._model_build,
                                          use_trial_id=-1)
            return self._lazymodel
        else:
            return self.study.best_trials

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
        if self.num_process > 1:
            self.trainer.state.fn = TrainerFn.FITTING  # add in lightning 1.6
            self.trainer.fit(*args, **kwargs)
        else:
            self.trainer._run(*args, **kwargs)
        self.trainer.tuning = True

    def _validate(self, *args: Any, **kwargs: Any) -> None:
        """A wrapper to test optimization latency multiple times after training."""
        self.trainer.validate_loop = CustomEvaluationLoop()
        self.trainer.state.fn = TrainerFn.VALIDATING
        self.trainer.training = False
        self.trainer.testing = False
        self.trainer.validate(*args, **kwargs)
