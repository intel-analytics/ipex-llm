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


from .objective import Objective
import copy
from .backend import PrunerType, SamplerType
from .backend import OptunaBackend

class HPOMixin:

    # argument keys for search, fit, tune creation, tune run.
    FIT_KEYS = (
        'x','y',
        'batch_size', 'epochs',
        'verbose','callbacks',
        'validation_split','validation_data',
        'shuffle','class_weight','sample_weight',
        'initial_epoch','steps_per_epoch',
        'validation_steps','validation_batch_size','validation_freq',
        'max_queue_size','workers','use_multiprocessing')

    TUNE_CREATE_KEYS = ('storage', 'sampler', 'sampler_kwargs',
                'pruner', 'pruner_kwargs', 'study_name', 'directions')


    TUNE_RUN_KEYS = ('timeout', 'n_jobs', 'catch', 'tune_callbacks',
                'gc_after_trial', 'show_progress_bar')

    # these methods are automatically created using "@proxy_methods"
    # details see desriptions in _proxy method
    PROXYED_METHODS = ['predict', 'predict_on_batch',
                       'evaluate', 'test_on_batch',
                       'to_json', 'to_yaml', 'summary',
                       'save', 'save_spec', 'save_weights',
                       'get_layer']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.objective = None
        self.study = None
        self.tune_end = False
        self._lazymodel = None


    def _strip_val_prefix(self,metric):
        if metric.startswith('val_'):
            metric = metric[len('val_'):]
        return metric

    def _check_optimize_direction(self, direction, metric):
        #TODO check common metrics and corresponding directions
        max_metrics=['accuracy','auc']
        min_metrics=['loss','mae','mse']
        stripped_metric = self._strip_val_prefix(metric).lower()
        if stripped_metric in max_metrics:
            if direction != 'maximize':
                raise ValueError('metric', metric,
                    'should use maximize direction for optmize')
        elif stripped_metric in min_metrics:
            if direction != 'minimize':
                raise ValueError('metric', metric,
                    'should use minimize direction for optmize')

    def _fix_target_metric(self, target_metric, fit_kwargs):
        compile_metrics=self.compile_kwargs.get('metrics',None)
        if target_metric is None:
            if fit_kwargs.get('validation_data', None) \
                or fit_kwargs.get('validation_split', None):
                    # if validation data or split is provided
                    # use validation metrics
                    prefix = 'val_'
            else:
                prefix = ''

            if compile_metrics is None:
                target_metric = prefix+'loss'
            elif isinstance(compile_metrics,list):
                target_metric = prefix+str(compile_metrics[0])
            else:
                target_metric = prefix+str(compile_metrics)
        elif isinstance(target_metric,list):
            raise ValueError("multiple objective metric is not supported.")
        else:
            stripped_target_metric = self._strip_val_prefix(target_metric)
            if compile_metrics is None:
                if stripped_target_metric not in ['loss','val_loss']:
                    raise ValueError("target metric is should be loss or val_loss",
                                     "if metrics is not provided in compile")
            elif isinstance(compile_metrics,list):
                if stripped_target_metric not in compile_metrics \
                    and stripped_target_metric not in ['loss','val_loss']:
                        raise ValueError("invalid target metric")
            else:
                if stripped_target_metric != compile_metrics \
                    and stripped_target_metric not in ['loss','val_loss']:
                        raise ValueError("invalid target metric")
        return target_metric

    def search(
        self,
        n_trails=1,
        resume=False,
        target_metric=None,
        direction="minimize",
        **kwargs
    ):
        """ Do the hyper param tuning.

        Args:
            n_trails (int, optional): number of trials to run. Defaults to 1.
            resume (bool, optional): whether to resume the previous tuning. Defaults to False.
            target_metric (str, optional): the target metric to optimize. Defaults to "accuracy".
            direction (str, optional): optimize direction. Defaults to "maximize".
            pruning (bool, optional): whether to use pruning
        """
        pruning = True if kwargs.get('pruner', None) else False

        ## create objective
        if self.objective is None:
            target_metric = self._fix_target_metric(target_metric, kwargs)
            fit_kwargs = self._filter_tuner_args(kwargs, HPOMixin.FIT_KEYS)
            self.objective = Objective(
                model=self._model_build,
                target_metric=target_metric,
                pruning = pruning,
                **fit_kwargs,
            )

        ## create study
        if self.study is None:
            if not resume:
                load_if_exists = False
                print("Starting a new tuning")
            else:
                load_if_exists = True
                print("Resume the last tuning...")

            study_create_kwargs = self._filter_tuner_args(kwargs, HPOMixin.TUNE_CREATE_KEYS)
            self._check_optimize_direction(direction,target_metric)

            # prepare sampler and pruner args

            sampler_type = study_create_kwargs.get('sampler', None)
            if sampler_type:
                sampler_args = study_create_kwargs.get('sampler_kwargs', {})
                sampler = OptunaBackend.create_sampler(sampler_type, sampler_args)
                study_create_kwargs['sampler'] = sampler
                study_create_kwargs.pop('sampler_kwargs',None)

            pruner_type = study_create_kwargs.get('pruner', None)
            if pruner_type:
                pruner_args=study_create_kwargs.get('pruner_kwargs', {})
                pruner = OptunaBackend.create_pruner(pruner_type, pruner_args)
                study_create_kwargs['pruner'] = pruner
                study_create_kwargs.pop('pruner_kwargs', None)

            self.study = OptunaBackend.create_study(
                direction=direction,
                load_if_exists=True,
                **study_create_kwargs
            )
            # self.study = optuna.create_study(
            #     direction=direction,
            #     load_if_exists=True,
            #     **study_create_kwargs)

        ## study optimize
        # rename callbacks to tune_callbacks to avoid conflict with fit param

        study_optimize_kwargs = self._filter_tuner_args(kwargs, HPOMixin.TUNE_RUN_KEYS)
        study_optimize_kwargs['callbacks'] = study_optimize_kwargs.get('tune_callbacks', None)
        study_optimize_kwargs.pop('tune_callbacks', None)
        self.study.optimize(
            self.objective, n_trials=n_trails, **study_optimize_kwargs)
        self.tune_end = False

    def search_summary(self):
        """Retrive a summary of trials

        Returns:
            dataframe: A summary of all the trials
        """
        if self.study is not None:
            print("Number of finished trials: {}".format(len(self.study.trials)))
            best = self.study.best_trial
            print("Best trial:")
            print("  Value: {}".format(best.value))
            print("  Params: ")
            for key, value in best.params.items():
                print("    {}: {}".format(key, value))
            return self.study
            # return self.study.trials_dataframe(attrs=("number", "value", "params", "state"))
        else:
            print("Seems you have not done any tuning yet.  \
                  Call tune and then call tune_summary to get the statistics.")

    def end_search(self, use_trial_id=-1):
        """ Put an end to tuning.
            Use the specified trial or best trial to init and
            compile the base model.

        Args:
            use_trial_id (int, optional): params of which trial to be used. Defaults to -1.

        Raises:
            ValueError: error when tune is not called already.
        """
        if self.objective is None or self.study is None:
            raise ValueError("Objective and study is not created.  \
                             Please call tune before calling end_tune. ")
        if use_trial_id == -1:
            trial = self.study.best_trial
        else:
            trial = self.study.trials[use_trial_id]

        self._lazymodel = self._model_build(trial)
        # TODO Next step: support retrive saved model instead of retrain from hparams
        self.tune_end = True

    def compile(self, *args, **kwargs):
        self.compile_args = args
        self.compile_kwargs = kwargs

    def fit(self, *args, **kwargs):
        if not self.tune_end:
            self.end_search()
        self._lazymodel.fit(*args, **kwargs)

    @staticmethod
    def _filter_tuner_args(kwargs, tuner_keys):
        return {k: v for k, v in kwargs.items() if k in tuner_keys}

    def _model_compile(self, model, trial):
        # for lazy model compile
        # TODO support searable compile args
        # config = OptunaBackend.sample_config(trial, kwspaces)
        # TODO objects like Optimizers has internal states so
        # each trial needs to have a copy of its own.
        # should allow users to pass a creator function
        # to avoid deep copy of objects
        compile_args = copy.deepcopy(self.compile_args)
        compile_kwargs = copy.deepcopy(self.compile_kwargs)
        model.compile(*compile_args, **compile_kwargs)

    def _model_build(self, trial):
        # for lazy model build
        # build model based on searched hyperparams from trial
        # TODO may add data creator here, e.g. refresh data, reset generators, etc.
        # super().__init__(**self._model_init_args(trial))
        # self._model_compile(super(), trial)
        # use composition instead of inherited
        #modelcls = self.__class__.__bases__[1]
        modelcls = self.model_class
        model = modelcls(**self._model_init_args(trial))
        #model = tf.keras.Model(**self._model_init_args(trial))
        self._model_compile(model, trial)
        return model

    def _proxy(self, name, method, *args, **kwargs):
        # call to keras method is forwarded to internal model
        # NOTE: keep the unused "method" argument so that
        # only the methods which are actually called are created
        if not self._lazymodel:
            raise ValueError(
                "Model is not actually built yet. Please call \
                'end_search' before calling '" + name + "'")
        internal_m = getattr(self._lazymodel, name)
        return internal_m(*args, **kwargs)
