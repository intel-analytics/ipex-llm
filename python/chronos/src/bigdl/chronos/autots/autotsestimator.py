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

import types

from bigdl.orca.automl.auto_estimator import AutoEstimator
from bigdl.chronos.data import TSDataset
import bigdl.orca.automl.hp as hp
from bigdl.chronos.autots.model import AutoModelFactory
from bigdl.chronos.autots.tspipeline import TSPipeline
from bigdl.chronos.autots.utils import recalculate_n_sampling


class AutoTSEstimator:
    """
    Automated TimeSeries Estimator for time series forecasting task, which supports
    TSDataset and customized data creator as data input on built-in model (only
    "lstm", "tcn", "seq2seq" for now) or 3rd party model.

    >>> # Here is a use case example:
    >>> # prepare train/valid/test tsdataset
    >>> autoest = AutoTSEstimator(model="lstm",
    >>>                           search_space=search_space,
    >>>                           past_seq_len=6,
    >>>                           future_seq_len=1)
    >>> tsppl = autoest.fit(data=tsdata_train,
    >>>                     validation_data=tsdata_valid)
    >>> tsppl.predict(tsdata_test)
    >>> tsppl.save("my_tsppl")
    """

    def __init__(self,
                 model="lstm",
                 search_space=dict(),
                 metric="mse",
                 metric_mode=None,
                 loss=None,
                 optimizer="Adam",
                 past_seq_len='auto',
                 future_seq_len=1,
                 input_feature_num=None,
                 output_target_num=None,
                 selected_features="auto",
                 backend="torch",
                 logs_dir="/tmp/autots_estimator",
                 cpus_per_trial=1,
                 name="autots_estimator",
                 remote_dir=None,
                 ):
        """
        AutoTSEstimator trains a model for time series forecasting.
        Users can choose one of the built-in models, or pass in a customized pytorch or keras model
        for tuning using AutoML.

        :param model: a string or a model creation function.
               A string indicates a built-in model, currently "lstm", "tcn", "seq2seq" are
               supported.
               A model creation function indicates a 3rd party model, the function should take a
               config param and return a torch.nn.Module (backend="torch") / tf model
               (backend="keras").
               If you use chronos.data.TSDataset as data input, the 3rd party
               should have 3 dim input (num_sample, past_seq_len, input_feature_num) and 3 dim
               output (num_sample, future_seq_len, output_feature_num) and use the same key
               in the model creation function. If you use a customized data creator, the output of
               data creator should fit the input of model creation function.
        :param search_space: str or dict. hyper parameter configurations. For str, you can choose
               from "minimal", "normal", or "large", each represents a default search_space for
               our built-in model with different computing requirement. For dict, Read the API docs
               for each auto model. Some common hyper parameter can be explicitly set in named
               parameter. search_space should contain those parameters other than the keyword
               arguments in this constructor in its key. If a 3rd parth model is used, then you
               must set search_space to a dict.
        :param metric: String or customized evaluation metric function.
               If string, metric is the evaluation metric name to optimize, e.g. "mse".
               If callable function, it signature should be func(y_true, y_pred), where y_true and
               y_pred are numpy ndarray. The function should return a float value as evaluation
               result.
        :param metric_mode: One of ["min", "max"]. "max" means greater metric value is better.
               You have to specify metric_mode if you use a customized metric function.
               You don't have to specify metric_mode if you use the built-in metric in
               bigdl.orca.automl.metrics.Evaluator.
        :param loss: String or pytorch loss instance or pytorch loss creator function. The
               default loss function for pytorch backend is nn.MSELoss(). If users use
               backend="keras" and 3rd parth model this parameter will be ignored.
        :param optimizer: String or pyTorch optimizer creator function or tf.keras optimizer
               instance. If users use backend="keras" and 3rd parth model, this parameter will
               be ignored.
        :param past_seq_len: Int or or hp sampling function. The number of historical steps (i.e.
               lookback) used for forecasting. For hp sampling, see bigdl.orca.automl.hp for more
               details. The values defaults to 'auto', which will automatically infer the
               cycle length of each time series and take the mode of them. The search space
               will be automatically set to hp.randint(0.5*cycle_length, 2*cycle_length).
        :param future_seq_len: Int or List. The number of future steps to forecast. The value
               defaults to 1, if `future_seq_len` is a list, we will sample discretely according
               to the input list. 1 means the timestamp just after the observed data.
        :param input_feature_num: Int. The number of features in the input. The value is ignored if
               you use chronos.data.TSDataset as input data type.
        :param output_target_num: Int. The number of targets in the output. The value is ignored if
               you use chronos.data.TSDataset as input data type.
        :param selected_features: String. "all" and "auto" are supported for now. For "all",
               all features that are generated are used for each trial. For "auto", a subset
               is sampled randomly from all features for each trial. The parameter is ignored
               if not using chronos.data.TSDataset as input data type. The value defaults
               to "auto".
        :param backend: The backend of the auto model. We only support backend as "torch" or
                      "keras" for now.
        :param logs_dir: Local directory to save logs and results.
               It defaults to "/tmp/autots_estimator"
        :param cpus_per_trial: Int. Number of cpus for each trial. It defaults to 1.
        :param name: name of the autots estimator. It defaults to "autots_estimator".
        :param remote_dir: String. Remote directory to sync training results and checkpoints. It
               defaults to None and doesn't take effects while running in local. While running in
               cluster, it defaults to "hdfs:///tmp/{name}".
        """
        from bigdl.nano.utils.log4Error import invalidInputError

        # check backend and set default loss MSE
        if backend == "torch":
            import torch
            if loss is None:
                loss = torch.nn.MSELoss()

        if isinstance(search_space, str):
            search_space = AutoModelFactory.get_default_search_space(model, search_space)

        self._future_seq_len = future_seq_len  # for support future_seq_len list input.
        invalidInputError(isinstance(future_seq_len, int) or isinstance(future_seq_len, list),
                          f"future_seq_len only support int or List, but found"
                          f" {type(future_seq_len)}")
        future_seq_len = future_seq_len if isinstance(future_seq_len, int) else len(future_seq_len)

        # 3rd party model
        if isinstance(model, types.FunctionType):
            from bigdl.orca.automl.auto_estimator import AutoEstimator
            if backend == "torch":
                self.model = AutoEstimator.from_torch(model_creator=model,
                                                      optimizer=optimizer,
                                                      loss=loss,
                                                      logs_dir=logs_dir,
                                                      resources_per_trial={"cpu": cpus_per_trial},
                                                      name=name)
            if backend == "keras":
                self.model = AutoEstimator.from_keras(model_creator=model,
                                                      logs_dir=logs_dir,
                                                      resources_per_trial={"cpu": cpus_per_trial},
                                                      name=name)
            self.metric = metric
            self.metric_mode = metric_mode
            search_space.update({"past_seq_len": past_seq_len,
                                 "future_seq_len": future_seq_len,
                                 "input_feature_num": input_feature_num,
                                 "output_feature_num": output_target_num})
            self.search_space = search_space

        # built-in model
        if isinstance(model, str):
            # update auto model common search space
            search_space.update({"past_seq_len": past_seq_len,
                                 "future_seq_len": future_seq_len,
                                 "input_feature_num": input_feature_num,
                                 "output_target_num": output_target_num,
                                 "loss": loss,
                                 "metric": metric,
                                 "metric_mode": metric_mode,
                                 "optimizer": optimizer,
                                 "backend": backend,
                                 "logs_dir": logs_dir,
                                 "cpus_per_trial": cpus_per_trial,
                                 "name": name})

            # create auto model from name
            self.model = AutoModelFactory.create_auto_model(name=model,
                                                            search_space=search_space)

        # save selected features setting for data creator generation
        self.selected_features = selected_features
        self.backend = backend
        self._scaler = None
        self._scaler_index = None

    def fit(self,
            data,
            epochs=1,
            batch_size=32,
            validation_data=None,
            metric_threshold=None,
            n_sampling=1,
            search_alg=None,
            search_alg_params=None,
            scheduler=None,
            scheduler_params=None
            ):
        """
        fit using AutoEstimator

        :param data: train data.
               For backend of "torch", data can be a TSDataset or a function that takes a
               config dictionary as parameter and returns a PyTorch DataLoader.

               For backend of "keras", data can be a TSDataset or a function that takes a
               config dictionary as parameter and returns a Tensorflow Dataset.

               Please notice that you should stick to the same data type when you
               predict/evaluate/fit on the TSPipeline you get from `AutoTSEstimator.fit`.
        :param epochs: Max number of epochs to train in each trial. Defaults to 1.
               If you have also set metric_threshold, a trial will stop if either it has been
               optimized to the metric_threshold or it has been trained for {epochs} epochs.
        :param batch_size: Int or hp sampling function from an integer space. Training batch size.
               It defaults to 32.
        :param validation_data: Validation data. Validation data type should be the same as data.
        :param metric_threshold: a trial will be terminated when metric threshold is met.
        :param n_sampling: Number of trials to evaluate in total. Defaults to 1.
               If hp.grid_search is in search_space, the grid will be run n_sampling of trials
               and round up n_sampling according to hp.grid_search.
               If this is -1, (virtually) infinite samples are generated
               until a stopping condition is met.
        :param search_alg: str, all supported searcher provided by ray tune
               (i.e."variant_generator", "random", "ax", "dragonfly", "skopt",
               "hyperopt", "bayesopt", "bohb", "nevergrad", "optuna", "zoopt" and
               "sigopt")
        :param search_alg_params: extra parameters for searcher algorithm besides search_space,
               metric and searcher mode
        :param scheduler: str, all supported scheduler provided by ray tune
        :param scheduler_params: parameters for scheduler

        :return: a TSPipeline with the best model.
        """
        is_third_party_model = isinstance(self.model, AutoEstimator)

        # generate data creator from TSDataset (pytorch base require validation data)
        if isinstance(data, TSDataset) and isinstance(validation_data, TSDataset):
            train_d, val_d = self._prepare_data_creator(
                search_space=self.search_space if is_third_party_model else self.model.search_space,
                train_data=data,
                val_data=validation_data,
            )
            self._scaler = data.scaler
            self._scaler_index = data.scaler_index
        else:
            train_d, val_d = data, validation_data

        if is_third_party_model:
            self.search_space.update({"batch_size": batch_size})
            n_sampling = recalculate_n_sampling(self.search_space,
                                                n_sampling) if n_sampling != -1 else -1
            self.model.fit(
                data=train_d,
                epochs=epochs,
                validation_data=val_d,
                metric=self.metric,
                metric_mode=self.metric_mode,
                metric_threshold=metric_threshold,
                n_sampling=n_sampling,
                search_space=self.search_space,
                search_alg=search_alg,
                search_alg_params=search_alg_params,
                scheduler=scheduler,
                scheduler_params=scheduler_params,
            )

        if not is_third_party_model:
            self.model.fit(
                data=train_d,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=val_d,
                metric_threshold=metric_threshold,
                n_sampling=n_sampling,
                search_alg=search_alg,
                search_alg_params=search_alg_params,
                scheduler=scheduler,
                scheduler_params=scheduler_params
            )

        if self.backend == "torch":
            best_model = self._get_best_automl_model()
            return TSPipeline(model=best_model.model,
                              loss=best_model.criterion,
                              optimizer=best_model.optimizer,
                              model_creator=best_model.model_creator,
                              loss_creator=best_model.loss_creator,
                              optimizer_creator=best_model.optimizer_creator,
                              best_config=self.get_best_config(),
                              scaler=self._scaler,
                              scaler_index=self._scaler_index)

        if self.backend == "keras":
            best_model = self._get_best_automl_model()
            return best_model

    def _prepare_data_creator(self, search_space, train_data, val_data=None):
        """
        prepare the data creators and add selected features to search_space
        :param search_space: the search space
        :param train_data: train data
        :param val_data: validation data
        :return: data creators from train and validation data
        """
        import ray
        from bigdl.nano.utils.log4Error import invalidInputError

        # automatically inference output_feature_num
        # input_feature_num will be set by base pytorch model according to selected features.
        search_space['output_feature_num'] = len(train_data.target_col)
        if search_space['past_seq_len'] == 'auto':
            cycle_length = train_data.get_cycle_length(aggregate='mode', top_k=3)
            cycle_length = 2 if cycle_length < 2 else cycle_length
            search_space['past_seq_len'] = hp.randint(cycle_length//2, cycle_length*2)

        # append feature selection into search space
        # TODO: more flexible setting
        all_features = train_data.feature_col
        if self.selected_features not in ('all', 'auto'):
            invalidInputError(False, "Only 'all' and 'auto' are supported for selected_features, "
                                     f"but found {self.selected_features}")
        if self.selected_features == "auto":
            if len(all_features) == 0:
                search_space['selected_features'] = all_features
            else:
                search_space['selected_features'] = hp.choice_n(all_features,
                                                                min_items=0,
                                                                max_items=len(all_features))
        if self.selected_features == "all":
            search_space['selected_features'] = all_features

        # put train/val data in ray
        train_data_id = ray.put(train_data)
        valid_data_id = ray.put(val_data)

        if self.backend == "torch":
            import torch
            from torch.utils.data import TensorDataset, DataLoader

            def train_data_creator(config):
                train_d = ray.get(train_data_id)

                x, y = train_d.roll(lookback=config.get('past_seq_len'),
                                    horizon=self._future_seq_len,
                                    feature_col=config['selected_features']) \
                              .to_numpy()

                return DataLoader(TensorDataset(torch.from_numpy(x).float(),
                                                torch.from_numpy(y).float()),
                                  batch_size=config["batch_size"],
                                  shuffle=True)

            def val_data_creator(config):
                val_d = ray.get(valid_data_id)

                x, y = val_d.roll(lookback=config.get('past_seq_len'),
                                  horizon=self._future_seq_len,
                                  feature_col=config['selected_features']) \
                            .to_numpy()

                return DataLoader(TensorDataset(torch.from_numpy(x).float(),
                                                torch.from_numpy(y).float()),
                                  batch_size=config["batch_size"],
                                  shuffle=True)

            return train_data_creator, val_data_creator

        if self.backend == "keras":
            def train_data_creator(config):
                train_d = ray.get(train_data_id)

                train_d.roll(lookback=config.get('past_seq_len'),
                             horizon=self._future_seq_len,
                             feature_col=config['selected_features'])

                return train_d.to_tf_dataset(batch_size=config["batch_size"],
                                             shuffle=True)

            def val_data_creator(config):
                val_d = ray.get(valid_data_id)

                val_d.roll(lookback=config.get('past_seq_len'),
                           horizon=self._future_seq_len,
                           feature_col=config['selected_features'])

                return val_d.to_tf_dataset(batch_size=config["batch_size"],
                                           shuffle=False)

            return train_data_creator, val_data_creator

    def _get_best_automl_model(self):
        """
        For internal use only.

        :return: the best automl model instance
        """
        return self.model._get_best_automl_model()

    def get_best_config(self):
        """
        Get the best configuration

        :return: A dictionary of best hyper parameters
        """
        return self.model.get_best_config()
