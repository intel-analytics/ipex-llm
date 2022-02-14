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
import optuna


class HPOMixin:

    def search(
        self,
        n_trails=1,
        resume=False,
        target_metric="accuracy",
        direction="maximize",
        **kwargs
    ):
        """ Do the hyper param tuning.

        Args:
            n_trails (int, optional): number of trials to run. Defaults to 1.
            resume (bool, optional): whether to resume the previous tuning. Defaults to False.
            target_metric (str, optional): the target metric to optimize. Defaults to "accuracy".
            direction (str, optional): optimize direction. Defaults to "maximize".
        """
        if self.objective is None:
            self.objective = Objective(
                model_cls=self.__class__.__bases__[1],
                model_initor=self._model_init_args,
                model_compiler=self._model_compile,
                target_metric=target_metric,
                **kwargs,
            )

        if self.study is None:
            if not resume:
                load_if_exists=False
                print("Starting a new tuning")
            else:
                load_if_exists=True
                print("Resume the last tuning...")
            create_keys = {'storage', 'sampler', 'pruner', 'study_name','directions'}
            create_kwargs = self._filter_tuner_args(kwargs, create_keys)
            self.study = optuna.create_study(direction=direction,  load_if_exists=True, **create_kwargs)
        optimize_keys = {'timeout', 'n_jobs', 'catch', 'callbacks', 'gc_after_trial', 'show_progress_bar'}
        optimize_kwargs = self._filter_tuner_args(kwargs, optimize_keys)
        self.study.optimize(self.objective, n_trials=n_trails, **optimize_kwargs)
        self.tune_end=False

    @staticmethod
    def _filter_tuner_args(kwargs, tuner_keys):
        return {k: v for k, v in kwargs.items() if k in tuner_keys}

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
            return self.study.trials_dataframe(attrs=("number", "value", "params", "state"))
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
        # TODO may add data creator here, e.g. refresh data, reset generators, etc.
        super().__init__(**self._model_init_args(trial))
        self._model_compile(super(), trial)
        #super().build()
        # TODO Next step: support retrive saved model instead of retrain from hparams
        self.tune_end=True
