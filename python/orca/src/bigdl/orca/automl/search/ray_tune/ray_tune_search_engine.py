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

import ray
from ray import tune
import os
from bigdl.orca.automl.search.base import SearchEngine, TrialOutput, GoodError
from bigdl.orca.automl.search.parameters import DEFAULT_LOGGER_NAME, DEFAULT_METRIC_NAME
from bigdl.orca.automl.search.ray_tune.utils import convert_bayes_configs
from bigdl.orca.automl.search.utils import get_ckpt_hdfs, put_ckpt_hdfs
from ray.tune import Stopper
from bigdl.orca.automl.model.abstract import ModelBuilder
from bigdl.orca.data import ray_xshards
from ray.tune.progress_reporter import TrialProgressCallback
from bigdl.dllib.utils.log4Error import *


class RayTuneSearchEngine(SearchEngine):
    """
    Tune driver
    """

    def __init__(self,
                 logs_dir="",
                 resources_per_trial=None,
                 name="",
                 remote_dir=None,
                 ):
        """
        Constructor
        :param logs_dir: local dir to save training results
        :param resources_per_trial: resources for each trial
        :param name: searcher name
        :param remote_dir: checkpoint will be uploaded to remote_dir in hdfs
        """
        self.train_func = None
        self.resources_per_trial = resources_per_trial
        self.trials = None
        self.name = name
        self.remote_dir = remote_dir or RayTuneSearchEngine.get_default_remote_dir(name)
        self.logs_dir = os.path.abspath(os.path.expanduser(logs_dir))

    @staticmethod
    def get_default_remote_dir(name):
        from bigdl.orca.ray import OrcaRayContext
        from bigdl.orca.automl.search.utils import process
        ray_ctx = OrcaRayContext.get()
        if ray_ctx.is_local:
            return None
        else:
            default_remote_dir = f"hdfs:///tmp/{name}"
            process(command=f"hadoop fs -mkdir -p {default_remote_dir}; "
                            f"hadoop fs -chmod 777 {default_remote_dir}")
            return default_remote_dir

    def compile(self,
                data,
                model_builder,
                metric_mode,
                epochs=1,
                validation_data=None,
                metric=None,
                metric_threshold=None,
                n_sampling=1,
                search_space=None,
                search_alg=None,
                search_alg_params=None,
                scheduler=None,
                scheduler_params=None,
                mc=False,
                feature_cols=None,
                label_cols=None,
                ):
        """
        Do necessary preparations for the engine
        :param data: data for training
               Pandas Dataframe:
                   a Pandas dataframe for training
               Numpy ndarray:
                   a tuple in form of (x, y)
                        x: ndarray for training input
                        y: ndarray for training output
               Spark Dataframe:
                   a Spark Dataframe for training
        :param model_builder: model creation function
        :param epochs: max epochs for training
        :param validation_data: validation data
        :param metric: metric name or metric function
        :param metric_mode: mode for metric. "min" or "max". We would infer metric_mode automated
            if user used our built-in metric in bigdl.automl.common.metric.Evaluator.
        :param metric_threshold: a trial will be terminated when metric threshold is met
        :param n_sampling: number of sampling
        :param search_space: a dictionary of search_space
        :param search_alg: str, all supported searcher provided by ray tune
               (i.e."variant_generator", "random", "ax", "dragonfly", "skopt",
               "hyperopt", "bayesopt", "bohb", "nevergrad", "optuna", "zoopt" and
               "sigopt")
        :param search_alg_params: extra parameters for searcher algorithm
        :param scheduler: str, all supported scheduler provided by ray tune
        :param scheduler_params: parameters for scheduler
        :param mc: if calculate uncertainty
        :param feature_cols: feature column names if data is Spark DataFrame.
        :param label_cols: target column names if data is Spark DataFrame.
        """
        # metric and metric's mode
        self.metric_name = metric.__name__ if callable(metric) else (metric or DEFAULT_METRIC_NAME)
        self.mode = metric_mode
        self.stopper = TrialStopper(metric_threshold=metric_threshold,
                                    epochs=epochs,
                                    metric=self.metric_name,
                                    mode=self.mode)
        self.num_samples = n_sampling
        self.search_space = search_space

        self._search_alg = RayTuneSearchEngine._set_search_alg(search_alg, search_alg_params,
                                                               self.metric_name, self.mode)
        self._scheduler = RayTuneSearchEngine._set_scheduler(scheduler, scheduler_params,
                                                             self.metric_name, self.mode)
        metric_func = None if not callable(metric) else metric
        self.train_func = self._prepare_train_func(data=data,
                                                   model_builder=model_builder,
                                                   validation_data=validation_data,
                                                   metric_name=self.metric_name,
                                                   metric_func=metric_func,
                                                   mode=self.mode,
                                                   mc=mc,
                                                   remote_dir=self.remote_dir,
                                                   resources_per_trial=self.resources_per_trial,
                                                   feature_cols=feature_cols,
                                                   label_cols=label_cols,
                                                   )

    @staticmethod
    def _set_search_alg(search_alg, search_alg_params, metric, mode):
        if search_alg:
            if not isinstance(search_alg, str):
                invalidInputError(False,
                                  f"search_alg should be of type str."
                                  f" Got {search_alg.__class__.__name__}")
            params = search_alg_params.copy() if search_alg_params else dict()
            if metric and "metric" not in params:
                params["metric"] = metric
            if mode and "mode" not in params:
                params["mode"] = mode
            search_alg = tune.create_searcher(search_alg, **params)
        return search_alg

    @staticmethod
    def _set_scheduler(scheduler, scheduler_params, metric, mode):
        if scheduler:
            if not isinstance(scheduler, str):
                invalidInputError(False,
                                  f"Scheduler should be of type str. "
                                  f"Got {scheduler.__class__.__name__}")
            params = scheduler_params.copy() if scheduler_params else dict()
            if metric and "metric" not in params:
                params["metric"] = metric
            if mode and "mode" not in params:
                params["mode"] = mode
            if "time_attr" not in params:
                params["time_attr"] = "training_iteration"
            scheduler = tune.create_scheduler(scheduler, **params)
        return scheduler

    def run(self):
        """
        Run trials
        :return: trials result
        """
        metric = self.metric_name if not self._scheduler else None
        mode = self.mode if not self._scheduler else None
        analysis = tune.run(
            self.train_func,
            local_dir=self.logs_dir,
            metric=metric,
            mode=mode,
            name=self.name,
            stop=self.stopper,
            config=self.search_space,
            search_alg=self._search_alg,
            num_samples=self.num_samples,
            trial_dirname_creator=trial_dirname_creator,
            callbacks=[CustomProgressCallback()],
            scheduler=self._scheduler,
            resources_per_trial=self.resources_per_trial,
            verbose=3,
            reuse_actors=True
        )
        self.trials = analysis.trials

        # Visualization code for ray (leaderboard)
        try:
            from bigdl.orca.automl.search import TensorboardLogger
            logger_name = self.name if self.name else DEFAULT_LOGGER_NAME
            tf_config, tf_metric = TensorboardLogger._ray_tune_searcher_log_adapt(analysis)

            self.logger = TensorboardLogger(logs_dir=os.path.join(self.logs_dir,
                                                                  logger_name+"_leaderboard"),
                                            name=logger_name)
            self.logger.run(tf_config, tf_metric)
            self.logger.close()
        except ImportError:
            import warnings
            warnings.warn("torch >= 1.7.0 should be installed to enable the orca.automl logger")

        return analysis

    def get_best_trial(self):
        return self.get_best_trials(k=1)[0]

    def get_best_trials(self, k=1):
        """
        get a list of best k trials
        :params k: top k
        :return: trials list
        """
        sorted_trials = RayTuneSearchEngine._get_sorted_trials(self.trials,
                                                               metric=self.metric_name,
                                                               mode=self.mode)
        best_trials = sorted_trials[:k]
        return [self._make_trial_output(t) for t in best_trials]

    def _make_trial_output(self, trial):
        model_path = os.path.join(trial.logdir, trial.last_result["checkpoint"])
        if self.remote_dir:
            get_ckpt_hdfs(self.remote_dir, model_path)
        return TrialOutput(config=trial.config,
                           model_path=model_path)

    @staticmethod
    def _get_best_trial(trial_list, metric, mode):
        """Retrieve the best trial."""
        if mode == "max":
            return max(trial_list, key=lambda trial: trial.last_result.get(metric, 0))
        else:
            return min(trial_list, key=lambda trial: trial.last_result.get(metric, 0))

    @staticmethod
    def _get_sorted_trials(trial_list, metric, mode):
        return sorted(
            trial_list,
            key=lambda trial: trial.last_result.get(metric, 0),
            reverse=True if mode == "max" else False)

    @staticmethod
    def _get_best_result(trial_list, metric, mode):
        """Retrieve the last result from the best trial."""
        return {metric: RayTuneSearchEngine._get_best_trial(trial_list,
                                                            metric, mode).last_result[metric]}

    def test_run(self):
        def mock_reporter(**kwargs):
            invalidInputError(self.metric_name in kwargs, "Did not report proper metric")
            invalidInputError("checkpoint" in kwargs, "Accidentally removed `checkpoint`?")
            invalidInputError(False,
                              "This works.")

        try:
            self.train_func({'out_units': 1,
                             'selected_features': ["MONTH(datetime)", "WEEKDAY(datetime)"]},
                            mock_reporter)
            # self.train_func(self.search_space, mock_reporter)

        except TypeError as e:
            print("Forgot to modify function signature?")
            invalidOperationError(False, str(e), cause=e)
        except GoodError:
            print("Works!")
            return 1
        invalidInputError(False, "Didn't call reporter...")

    @staticmethod
    def _prepare_train_func(data,
                            model_builder,
                            validation_data=None,
                            metric_name=None,
                            metric_func=None,
                            mode=None,
                            mc=False,
                            remote_dir=None,
                            resources_per_trial=None,
                            feature_cols=None,
                            label_cols=None,
                            ):
        """
        Prepare the train function for ray tune
        :param data: input data
        :param model_builder: model create function
        :param metric_name: the rewarding metric name
        :param metric_func: customized metric func
        :param mode: metric mode
        :param validation_data: validation data
        :param mc: if calculate uncertainty
        :param remote_dir: checkpoint will be uploaded to remote_dir in hdfs

        :return: the train function
        """
        from pyspark.sql import DataFrame
        if isinstance(data, DataFrame):
            from bigdl.orca.learn.utils import dataframe_to_xshards
            from bigdl.dllib.utils.common import get_node_and_core_number
            from bigdl.orca.data.utils import process_spark_xshards
            num_workers, _ = get_node_and_core_number()
            spark_xshards, val_spark_xshards = dataframe_to_xshards(data,
                                                                    validation_data=validation_data,
                                                                    feature_cols=feature_cols,
                                                                    label_cols=label_cols,
                                                                    mode="fit",
                                                                    num_workers=num_workers)
            ray_xshards = process_spark_xshards(spark_xshards, num_workers=num_workers)
            val_ray_xshards = process_spark_xshards(val_spark_xshards, num_workers=num_workers)
            data_ref = ray_xshards.get_partition_refs()
            validation_data_ref = val_ray_xshards.get_partition_refs()
        else:
            data_ref = ray.put(data)
            validation_data_ref = ray.put(validation_data)

        def train_func(config):
            if isinstance(data_ref, list):
                train_data = get_data_from_part_refs(data_ref)
                val_data = get_data_from_part_refs(validation_data_ref)
            else:
                train_data = ray.get(data_ref)
                val_data = ray.get(validation_data_ref)
            config = convert_bayes_configs(config).copy()
            trial_model = model_builder.build(config)

            # no need to call build since it is called the first time fit_eval is called.
            # callbacks = [TuneCallback(tune_reporter)]
            # fit model
            best_reward = None
            for i in range(1, 101):
                result = trial_model.fit_eval(data=train_data,
                                              validation_data=val_data,
                                              mc=mc,
                                              metric=metric_name,
                                              metric_func=metric_func,
                                              resources_per_trial=resources_per_trial,
                                              **config)
                reward = result[metric_name]
                checkpoint_filename = "best.ckpt"

                # Save best reward iteration
                if mode == "max":
                    has_best_reward = best_reward is None or reward > best_reward
                elif mode == "min":
                    has_best_reward = best_reward is None or reward < best_reward
                else:
                    has_best_reward = True

                if has_best_reward:
                    best_reward = reward
                    trial_model.save(checkpoint_filename)
                    # Save to hdfs
                    if remote_dir is not None:
                        put_ckpt_hdfs(remote_dir, checkpoint_filename)

                report_dict = {"training_iteration": i,
                               "checkpoint": checkpoint_filename,
                               "best_" + metric_name: best_reward}
                report_dict.update(result)
                tune.report(**report_dict)

        return train_func


# stopper
class TrialStopper(Stopper):
    def __init__(self, metric_threshold, epochs, metric, mode):
        self._mode = mode
        self._metric = metric
        self._metric_threshold = metric_threshold
        self._epochs = epochs

    def __call__(self, trial_id, result):
        if self._metric_threshold is not None:
            if self._mode == "max" and result[self._metric] >= self._metric_threshold:
                return True
            if self._mode == "min" and result[self._metric] <= self._metric_threshold:
                return True
        if result["training_iteration"] >= self._epochs:
            return True
        return False

    def stop_all(self):
        return False


class CustomProgressCallback(TrialProgressCallback):
    # for a clearer log in tuning
    def log_result(self, trial, result, error: bool = False):
        pass


def trial_dirname_creator(trial):
    return f"{trial.trainable_name}_{trial.trial_id}"


def get_data_from_part_refs(part_refs):
    from bigdl.orca.data.utils import ray_partitions_get_data_label

    partitions = [ray.get(part_ref) for part_ref in part_refs]

    data, label = ray_partitions_get_data_label(partitions,
                                                allow_tuple=True,
                                                allow_list=False,
                                                )
    return data, label
