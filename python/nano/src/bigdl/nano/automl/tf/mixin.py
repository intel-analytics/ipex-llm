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

import copy
import warnings

from .objective import Objective
from bigdl.nano.automl.hpo.backend import create_hpo_backend
from bigdl.nano.automl.hpo.search import (
    _search_summary,
    _end_search,
    _create_study,
    _validate_args,
    _strip_val_prefix,
    _prepare_args,
)
from bigdl.nano.automl.hpo.space import AutoObject
from bigdl.nano.utils.log4Error import invalidInputError
from bigdl.nano.automl.utils.parallel import run_parallel
import math


class HPOMixin:
    """
    A Minin object to add hpo related methods to models.

    It is used to add hpo related search methods onto
    tf.keras.Sequential, tf.keras.Model, as well as custom
        model subclassing from tf.keras.Model.
    """

    # argument keys for search, fit, tune creation, tune run.
    FIT_KEYS = {
        'x', 'y',
        'batch_size', 'epochs',
        'verbose', 'callbacks',
        'validation_split', 'validation_data',
        'shuffle', 'class_weight', 'sample_weight',
        'initial_epoch', 'steps_per_epoch',
        'validation_steps', 'validation_batch_size', 'validation_freq',
        'max_queue_size', 'workers', 'use_multiprocessing'}

    TUNE_CREATE_KEYS = {'storage', 'sampler', 'sampler_kwargs',
                        'pruner', 'pruner_kwargs', 'study_name', 'load_if_exists',
                        'direction', 'directions'}

    TUNE_RUN_KEYS = {'n_trials', 'timeout', 'n_jobs', 'catch', 'tune_callbacks',
                     'gc_after_trial', 'show_progress_bar'}

    # these methods are automatically created using "@proxy_methods"
    # details see desriptions in _proxy method
    PROXYED_METHODS = ['predict', 'predict_on_batch',
                       'evaluate', 'test_on_batch',
                       'to_json', 'to_yaml', 'summary',
                       'save', 'save_spec', 'save_weights',
                       'get_layer']

    def __init__(self, *args, **kwargs):
        """Init the Mixin."""
        super().__init__(*args, **kwargs)
        self.objective = None
        self.study = None
        self.tune_end = False
        self._lazymodel = None
        self.backend = create_hpo_backend()

    def _fix_target_metric(self, target_metric, fit_kwargs):
        compile_metrics = self.compile_kwargs.get('metrics', None)
        if target_metric is None:
            if fit_kwargs.get('validation_data', None) or fit_kwargs.get('validation_split', None):
                # if validation data or split is provided
                # use validation metrics
                prefix = 'val_'
            else:
                prefix = ''

            if compile_metrics is None:
                target_metric = prefix + 'loss'
            elif isinstance(compile_metrics, list):
                target_metric = prefix + str(compile_metrics[0])
            else:
                target_metric = prefix + str(compile_metrics)
        elif isinstance(target_metric, list):
            invalidInputError(False, "multiple objective metric is not supported.")
        else:
            stripped_target_metric = _strip_val_prefix(target_metric)
            if compile_metrics is None:
                if stripped_target_metric not in ['loss', 'val_loss']:
                    invalidInputError(False, "target metric is should be loss or val_loss"
                                             " if metrics is not provided in compile")
            elif isinstance(compile_metrics, list):
                target_not_in = stripped_target_metric not in ['loss', 'val_loss']
                if stripped_target_metric not in compile_metrics and target_not_in:
                    invalidInputError(False, "invalid target metric")
            else:
                target_not_in = stripped_target_metric not in ['loss', 'val_loss']
                if stripped_target_metric != compile_metrics and target_not_in:
                    invalidInputError(False, "invalid target metric")
        return target_metric

    @staticmethod
    def _create_objective(model_builder,
                          target_metric,
                          isprune,
                          fit_kwargs,
                          backend,
                          report_method):
        objective = Objective(
            model=model_builder,
            target_metric=target_metric,
            pruning=isprune,
            backend=backend,
            report_method=report_method,
            **fit_kwargs,
        )
        return objective

    def _get_model_builder_args(self):
        return {'model_init_args_func': self._model_init_args_func,
                'model_init_args_func_kwargs': self._get_model_init_args_func_kwargs(),
                'modelcls': self.model_class,
                'compile_args': self.compile_args,
                'compile_kwargs': self.compile_kwargs,
                'backend': self.backend}

    @staticmethod
    def _get_model_builder(model_init_args_func,
                           model_init_args_func_kwargs,
                           modelcls,
                           compile_args,
                           compile_kwargs,
                           backend):

        def model_builder(trial):
            model = modelcls(**model_init_args_func(
                trial,
                **model_init_args_func_kwargs))
            # self._model_compile(model, trial)
            # instantiate optimizers if it is autoobj
            optimizer = compile_kwargs.get('optimizer', None)
            if optimizer and isinstance(optimizer, AutoObject):
                optimizer = backend.instantiate(trial, optimizer)
                compile_kwargs['optimizer'] = optimizer
            model.compile(*compile_args, **compile_kwargs)
            return model
        return model_builder

    @staticmethod
    def _run_search_subproc(study,
                            get_model_builder_func,
                            get_model_builder_func_args,
                            backend,
                            target_metric,
                            isprune,
                            report_method,
                            fit_kwargs,
                            run_kwargs):
        """A stand-alone function for running parallel search."""
        # # run optimize
        model_builder = get_model_builder_func(**get_model_builder_func_args)

        objective = HPOMixin._create_objective(model_builder,
                                               target_metric,
                                               isprune,
                                               fit_kwargs,
                                               backend,
                                               report_method)

        study.optimize(objective, **run_kwargs)

    def _run_search_n_procs(self, isprune, n_procs=4):

        # subp_study = copy.deepcopy(self.study)
        # subp_objective = copy.deepcopy(self.objective)
        subp_run_kwargs = copy.deepcopy(self.run_kwargs)
        n_trials = subp_run_kwargs.get('n_trials', None)
        if n_trials:
            subp_n_trials = math.ceil(n_trials / n_procs)
            subp_run_kwargs['n_trials'] = subp_n_trials

        subp_kwargs = {'study': self.study,
                       'get_model_builder_func': self._get_model_builder,
                       'get_model_builder_func_args': self._get_model_builder_args(),
                       'backend': self.backend,
                       'target_metric': self.target_metric,
                       'isprune': isprune,
                       'report_method': self.report_method,
                       'fit_kwargs': self.fit_kwargs,
                       'run_kwargs': subp_run_kwargs}

        # set_loky_pickler('pickle')
        # Parallel(n_jobs=2)(_run_search(**subp_kwargs) for _ in range(1))
        # set_loky_pickler()
        run_parallel(func=self._run_search_subproc,
                     kwargs=subp_kwargs,
                     n_procs=n_procs)

    def _run_search_n_threads(self, isprune, n_threads=1):
        self.run_kwargs['n_jobs'] = n_threads
        self._run_search(isprune)

    def _run_search(self, isprune):
        if self.objective is None:
            self.objective = self._create_objective(self._model_build,
                                                    self.target_metric,
                                                    isprune,
                                                    self.fit_kwargs,
                                                    self.backend,
                                                    self.report_method)

        self.study.optimize(self.objective, **self.run_kwargs)

    @staticmethod
    def _prepare_report_method(mode, direction):
        if mode == 'auto':
            if direction == 'maximize':
                mode = 'max'
            else:
                mode = 'min'
        if mode == 'max':
            return max
        elif mode == 'min':
            return min
        elif mode == 'last':
            return lambda x: x[-1]
        else:
            invalidInputError(False, "mode is not recognized")

    def search(
        self,
        resume=False,
        target_metric=None,
        n_parallels=1,
        target_metric_mode='last',
        **kwargs
    ):
        """
        Do the hyper param search.

        :param resume: bool, optional. whether to resume the previous tuning.
            Defaults to False.
        :param target_metric: str, optional. the target metric to optimize.
            Defaults to "accuracy".
        :param n_parallels: number of parallel processes to run trials.
        :param target_metric_mode: target metric of which epoch to report as the final result,
            possible options are:
                'max': maximum value of all epochs's results
                'min': minimum value of all epochs's results
                'last': result of the last epoch
                'auto': if direction is maximize, use max mode
                        if direction is minimize, use min mode
        :param kwargs: model.fit arguments (e.g. batch_size, validation_data, etc.)
            and search backend arguments (e.g. n_trials, pruner, etc.)
            are allowed in kwargs.
        """
        do_create = True
        if resume:
            if 'storage' not in kwargs.keys() or kwargs['storage'].strip() == "":
                if self.study is None:
                    warnings.warn(
                        "A new study is created since there's no existing study to resume from.",
                        UserWarning)
                else:
                    do_create = False

        if 'storage' in kwargs.keys() and kwargs['storage'].strip() == "":
            del kwargs['storage']

        search_kwargs = kwargs or {}
        self.target_metric = self._fix_target_metric(target_metric, kwargs)

        _validate_args(search_kwargs,
                       self.target_metric,
                       legal_keys=[HPOMixin.FIT_KEYS,
                                   HPOMixin.TUNE_CREATE_KEYS,
                                   HPOMixin.TUNE_RUN_KEYS])

        (self.create_kwargs, self.run_kwargs, self.fit_kwargs) \
            = _prepare_args(search_kwargs,
                            HPOMixin.TUNE_CREATE_KEYS,
                            HPOMixin.TUNE_RUN_KEYS,
                            HPOMixin.FIT_KEYS,
                            self.backend)

        self.report_method = self._prepare_report_method(
            target_metric_mode, self.create_kwargs.get('direction', None))

        # create study
        if do_create:
            self.study = _create_study(resume, self.create_kwargs, self.backend)

        isprune = True if self.create_kwargs.get('pruner', None) else False
        if n_parallels and n_parallels > 1:
            # if storage is in-memory, use parallel threads instead of multi-process
            # this may suffer form python's GIL.
            if self.create_kwargs.get('storage', "").strip() == "":
                self._run_search_n_threads(isprune, n_threads=n_parallels)
            else:
                self._run_search_n_procs(isprune, n_procs=n_parallels)
        else:
            self._run_search(isprune)

        self.tune_end = False

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

    def compile(self, *args, **kwargs):
        """Collect compile arguments and delay it to each trial\
            and end_search."""
        self.compile_args = args
        self.compile_kwargs = kwargs

    def fit(self, *args, **kwargs):
        """Fit using the built-model form end_search."""
        if not self.tune_end:
            self.end_search()
        self._lazymodel.fit(*args, **kwargs)

    def _model_compile(self, model, trial):
        # for lazy model compile
        # objects like Optimizers has internal states so
        # each trial needs to have a copy of its own.
        # TODO may allow users to pass a creator function
        # to avoid deep copy of objects
        compile_args = copy.deepcopy(self.compile_args)
        compile_kwargs = copy.deepcopy(self.compile_kwargs)

        # instantiate optimizers if it is autoobj
        optimizer = compile_kwargs.get('optimizer', None)
        if optimizer and isinstance(optimizer, AutoObject):
            optimizer = self.backend.instantiate(trial, optimizer)
            compile_kwargs['optimizer'] = optimizer
        model.compile(*compile_args, **compile_kwargs)

    def _model_build(self, trial):
        # for lazy model build
        # build model based on searched hyperparams from trial
        # TODO may add data creator here, e.g. refresh data, reset generators, etc.
        # super().__init__(**self._model_init_args(trial))
        # self._model_compile(super(), trial)
        # use composition instead of inherited
        # modelcls = self.__class__.__bases__[1]
        modelcls = self.model_class
        model = modelcls(**self._model_init_args(trial))
        # model = tf.keras.Model(**self._model_init_args(trial))
        self._model_compile(model, trial)
        return model

    def _proxy(self, name, method, *args, **kwargs):
        # call to keras method is forwarded to internal model
        # NOTE: keep the unused "method" argument so that
        # only the methods which are actually called are created
        if not self._lazymodel:
            invalidInputError(False,
                              "Model is not actually built yet. Please call \
                              'end_search' before calling '" + name + "'")
        internal_m = getattr(self._lazymodel, name)
        return internal_m(*args, **kwargs)
