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

from bigdl.chronos.forecaster.abstract import Forecaster
from bigdl.chronos.forecaster.utils import *
from bigdl.chronos.metric.forecast_metrics import Evaluator

from typing import Optional
import numpy as np
import warnings
# Filter out useless Userwarnings
warnings.filterwarnings('ignore', category=UserWarning, module='pytorch_lightning')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
import os
from tempfile import TemporaryDirectory
import torch

from torch.utils.data import TensorDataset, DataLoader
from .utils_hpo import GenericLightningModule, _format_metric_str, _config_has_search_space
from bigdl.nano.utils.log4Error import invalidOperationError, invalidInputError
from bigdl.chronos.data.tsdataset import TSDataset


class BasePytorchForecaster(Forecaster):
    '''
    Forecaster base model for lstm, seq2seq, tcn and nbeats forecasters.
    '''
    def __init__(self, **kwargs):
        self.internal = None
        if self.distributed:
            # don't support use_hpo when distributed
            self.use_hpo = False
            from bigdl.orca.learn.pytorch.estimator import Estimator
            from bigdl.orca.learn.metrics import MSE, MAE
            ORCA_METRICS = {"mse": MSE, "mae": MAE}

            def model_creator_orca(config):
                set_pytorch_seed(self.seed)
                model = self.model_creator({**self.model_config, **self.data_config})
                model.train()
                return model
            self.internal = Estimator.from_torch(model=model_creator_orca,
                                                 optimizer=self.optimizer_creator,
                                                 loss=self.loss_creator,
                                                 metrics=[ORCA_METRICS[name]()
                                                          for name in self.metrics],
                                                 backend=self.remote_distributed_backend,
                                                 use_tqdm=True,
                                                 config={"lr": self.lr},
                                                 workers_per_node=self.workers_per_node)
        else:
            # seed setting
            from pytorch_lightning import seed_everything
            from bigdl.chronos.pytorch import TSTrainer as Trainer
            seed_everything(seed=self.seed)

            # Model preparation
            self.fitted = False

            has_space = _config_has_search_space(
                config={**self.model_config, **self.optim_config,
                        **self.loss_config, **self.data_config})

            if not self.use_hpo and has_space:
                invalidInputError(False, "Found search spaces in arguments but HPO is disabled."
                                         "Enable HPO or remove search spaces in arguments to use.")

            if not has_space:
                self.use_hpo = False
                model = self.model_creator({**self.model_config, **self.data_config})
                loss = self.loss_creator(self.loss_config)
                optimizer = self.optimizer_creator(model, self.optim_config)
                self.internal = Trainer.compile(model=model, loss=loss,
                                                optimizer=optimizer)

            self.accelerated_model = None  # accelerated model obtained from various accelerators
            self.accelerate_method = None  # str indicates current accelerate method

    def _build_automodel(self, data, validation_data=None, batch_size=32, epochs=1):
        """Build a Generic Model using config parameters."""
        merged_config = {**self.model_config, **self.optim_config,
                         **self.loss_config, **self.data_config}

        model_config_keys = list(self.model_config.keys())
        data_config_keys = list(self.data_config.keys())
        optim_config_keys = list(self.optim_config.keys())
        loss_config_keys = list(self.loss_config.keys())

        return GenericLightningModule(
            model_creator=self.model_creator,
            optim_creator=self.optimizer_creator,
            loss_creator=self.loss_creator,
            data=data, validation_data=validation_data,
            batch_size=batch_size, epochs=epochs,
            metrics=[_str2metric(metric) for metric in self.metrics],
            scheduler=None,  # TODO
            num_processes=self.num_processes,
            model_config_keys=model_config_keys,
            data_config_keys=data_config_keys,
            optim_config_keys=optim_config_keys,
            loss_config_keys=loss_config_keys,
            **merged_config)

    def tune(self,
             data,
             validation_data,
             target_metric,
             direction,
             directions=None,
             n_trials=2,
             n_parallels=1,
             epochs=1,
             batch_size=32,
             acceleration=False,
             input_sample=None,
             **kwargs):
        """
        Search the hyper parameter.

        :param data: train data, as numpy ndarray tuple (x, y)
        :param validation_data: validation data, as numpy ndarray tuple (x,y)
        :param target_metric: the target metric to optimize,
               a string or an instance of torchmetrics.metric.Metric
        :param direction: in which direction to optimize the target metric,
               "maximize" - larger the better
               "minimize" - smaller the better
        :param n_trials: number of trials to run
        :param n_parallels: number of parallel processes used to run trials.
               to use parallel tuning you need to use a RDB url for storage and specify study_name.
               For more information, refer to Nano AutoML user guide.
        :param epochs: the number of epochs to run in each trial fit, defaults to 1
        :param batch_size: number of batch size for each trial fit, defaults to 32
        :param acceleration: Whether to automatically consider the model after
            inference acceleration in the search process. It will only take
            effect if target_metric contains "latency". Default value is False.
        :param input_sample: A set of inputs for trace, defaults to None if you have
            trace before or model is a LightningModule with any dataloader attached.
        :param kwargs: some other parameters could be used for tuning, most useful one is
               `sampler` from SamplerType.Grid, SamplerType.Random and SamplerType.TPE so on.
        """

        invalidInputError(not self.distributed,
                          "HPO is not supported in distributed mode."
                          "Please use AutoTS instead.")
        invalidOperationError(self.use_hpo,
                              "HPO is disabled for this forecaster."
                              "You may specify search space in hyper parameters to enable it.")

        # prepare data
        from bigdl.chronos.pytorch import TSTrainer as Trainer

        # data transformation
        if isinstance(data, tuple):
            check_data(data[0], data[1], self.data_config)
            if validation_data and isinstance(validation_data, tuple):
                check_data(validation_data[0], validation_data[1], self.data_config)
            else:
                invalidInputError(False,
                                  "To use tuning, you must provide validation_data"
                                  "as numpy arrays.")
        else:
            invalidInputError(False, "HPO only supports numpy train input data.")

        if input_sample is None:
            input_sample = torch.from_numpy(data[0][:1, :, :])

        # prepare target metric
        if validation_data is not None:
            formated_target_metric = _format_metric_str('val', target_metric)
        else:
            invalidInputError(False, "To use tuning, you must provide validation_data"
                                     "as numpy arrays.")

        # build auto model
        self.tune_internal = self._build_automodel(data, validation_data, batch_size, epochs)

        # shall we use the same trainier
        self.tune_trainer = Trainer(logger=False, max_epochs=epochs,
                                    enable_checkpointing=self.checkpoint_callback,
                                    num_processes=self.num_processes, use_ipex=self.use_ipex,
                                    use_hpo=True)

        # run hyper parameter search
        self.internal = self.tune_trainer.search(
            self.tune_internal,
            n_trials=n_trials,
            target_metric=formated_target_metric,
            direction=direction,
            directions=directions,
            n_parallels=n_parallels,
            acceleration=acceleration,
            input_sample=input_sample,
            **kwargs)

        if self.tune_trainer.hposearcher.objective.mo_hpo:
            return self.internal
        # else:
            # reset train and validation datasets
            # self.tune_trainer.reset_train_val_dataloaders(self.internal)

    def search_summary(self):
        """
        Return search summary of HPO.
        """
        # add tuning check
        invalidOperationError(self.use_hpo, "No search summary when HPO is disabled.")
        return self.tune_trainer.search_summary()

    def fit(self, data, validation_data=None, epochs=1, batch_size=32, validation_mode='output',
            earlystop_patience=1, use_trial_id=None):
        """
        Fit(Train) the forecaster.

        :param data: The data support following formats:

               | 1. a numpy ndarray tuple (x, y):
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.
               |
               | 2. a xshard item:
               | each partition can be a dictionary of {'x': x, 'y': y}, where x and y's shape
               | should follow the shape stated before.
               |
               | 3. pytorch dataloader:
               | the dataloader should return x, y in each iteration with the shape as following:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.
               |
               | 4. A bigdl.chronos.data.tsdataset.TSDataset instance:
               | Forecaster will automatically process the TSDataset.
               | By default, TSDataset will be transformed to a pytorch dataloader,
               | which is memory-friendly while a little bit slower.
               | Users may call `roll` on the TSDataset before calling `fit`
               | Then the training speed will be faster but will consume more memory.

        :param validation_data: Validation sample for validation loop. Defaults to 'None'.
               If you do not input data for 'validation_data', the validation_step will be skipped.
               Validation data will be ignored under distributed mode.
               The validation_data support following formats:

               | 1. a numpy ndarray tuple (x, y):
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.
               |
               | 2. a xshard item:
               | each partition can be a dictionary of {'x': x, 'y': y}, where x and y's shape
               | should follow the shape stated before.
               |
               | 3. pytorch dataloader:
               | the dataloader should return x, y in each iteration with the shape as following:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.
               |
               | 4. A bigdl.chronos.data.tsdataset.TSDataset instance:
               | Forecaster will automatically process the TSDataset.
               | By default, TSDataset will be transformed to a pytorch dataloader,
               | which is memory-friendly while a little bit slower.
               | Users may call `roll` on the TSDataset before calling `fit`
               | Then the training speed will be faster but will consume more memory.

        :param epochs: Number of epochs you want to train. The value defaults to 1.
        :param batch_size: Number of batch size you want to train. The value defaults to 32.
               If you input a pytorch dataloader for `data`, the batch_size will follow the
               batch_size setted in `data`.if the forecaster is distributed, the batch_size will be
               evenly distributed to all workers.
        :param validation_mode:  A str represent the operation mode while having 'validation_data'.
               Defaults to 'output'. The validation_mode includes the following types:

               | 1. output:
               | If you choose 'output' for validation_mode, it will return a dict that records the
               | average validation loss of each epoch.
               |
               | 2. earlystop:
               | Monitor the val_loss and stop training when it stops improving.
               |
               | 3. best_epoch:
               | Monitor the val_loss. And load the checkpoint of the epoch with the smallest
               | val_loss after the training.

        :param earlystop_patience: Number of checks with no improvement after which training will
               be stopped. It takes effect when 'validation_mode' is 'earlystop'. Under the default
               configuration, one check happens after every training epoch.
        :param use_trail_id: choose a internal according to trial_id, which is used only
               in multi-objective search.
        :return: Validation loss if 'validation_data' is not None.
        """
        # input transform
        if isinstance(data, TSDataset):
            _rolled = data.numpy_x is None
            data = data.to_torch_data_loader(batch_size=batch_size,
                                             roll=_rolled,
                                             lookback=self.data_config['past_seq_len'],
                                             horizon=self.data_config['future_seq_len'],
                                             feature_col=data.roll_feature,
                                             target_col=data.roll_target,
                                             shuffle=True)
        if isinstance(data, DataLoader) and self.distributed:
            data = loader_to_creator(data)
        if isinstance(data, tuple) and self.distributed:
            data = np_to_creator(data)
        try:
            from bigdl.orca.data.shard import SparkXShards
            if isinstance(data, SparkXShards) and not self.distributed:
                warnings.warn("Xshards is collected to local since the "
                              "forecaster is non-distribued.")
                data = xshard_to_np(data)
            if isinstance(validation_data, SparkXShards) and not self.distributed:
                warnings.warn("Xshards is collected to local since the "
                              "forecaster is non-distribued.")
                validation_data = xshard_to_np(validation_data)
        except ImportError:
            pass

        invalidOperationError(self.internal is not None,
                              "The model is not properly built. "
                              "Have you set search spaces in arguments? "
                              "If so, you need to run tune before fit "
                              "to search and build the model.")

        # fit on internal
        if self.distributed:
            # for cluster mode
            from bigdl.orca.common import OrcaContext
            sc = OrcaContext.get_spark_context().getConf()
            num_nodes = 1 if sc.get('spark.master').startswith('local') \
                else int(sc.get('spark.executor.instances'))
            if batch_size % self.workers_per_node != 0:
                from bigdl.nano.utils.log4Error import invalidInputError
                invalidInputError(False,
                                  "Please make sure that batch_size can be divisible by "
                                  "the product of worker_per_node and num_nodes, "
                                  f"but 'batch_size' is {batch_size}, 'workers_per_node' "
                                  f"is {self.workers_per_node}, 'num_nodes' is {num_nodes}")
            batch_size //= (self.workers_per_node * num_nodes)
            return self.internal.fit(data=data,
                                     epochs=epochs,
                                     batch_size=batch_size)
        else:
            from bigdl.chronos.pytorch import TSTrainer as Trainer
            from bigdl.nano.utils.log4Error import invalidInputError

            # numpy data shape checking
            if isinstance(data, tuple):
                check_data(data[0], data[1], self.data_config)

            # data transformation
            if isinstance(data, tuple):
                data = np_to_dataloader(data, batch_size, self.num_processes)

            # training process
            # forecaster_log_dir is a temp directory for training log
            # validation_ckpt_dir is a temp directory for best checkpoint on validation data
            with TemporaryDirectory() as forecaster_log_dir:
                with TemporaryDirectory() as validation_ckpt_dir:
                    from pytorch_lightning.loggers import CSVLogger
                    logger = False if validation_data is None else CSVLogger(
                        save_dir=forecaster_log_dir,
                        flush_logs_every_n_steps=10,
                        name="forecaster_tmp_log")
                    from pytorch_lightning.callbacks import EarlyStopping
                    early_stopping = EarlyStopping('val/loss', patience=earlystop_patience)
                    from pytorch_lightning.callbacks import ModelCheckpoint
                    checkpoint_callback = ModelCheckpoint(monitor="val/loss",
                                                          dirpath=validation_ckpt_dir,
                                                          filename='best',
                                                          save_on_train_epoch_end=True)
                    if validation_mode == 'earlystop':
                        callbacks = [early_stopping]
                    elif validation_mode == 'best_epoch':
                        callbacks = [checkpoint_callback]
                    else:
                        callbacks = None
                    # Trainer init
                    self.trainer = Trainer(logger=logger, max_epochs=epochs, callbacks=callbacks,
                                           enable_checkpointing=self.checkpoint_callback,
                                           num_processes=self.num_processes, use_ipex=self.use_ipex,
                                           log_every_n_steps=10,
                                           distributed_backend=self.local_distributed_backend)

                    # This error is only triggered when the python
                    # interpreter starts additional processes.
                    # num_process=1 and subprocess will be safely started in the main process,
                    # so this error will not be triggered.
                    invalidInputError(is_main_process(),
                                      "Make sure new Python interpreters can "
                                      "safely import the main module. ",
                                      fixMsg="you should use if __name__ == '__main__':, "
                                      "otherwise performance will be degraded.")

                    # build internal according to use_trail_id for multi-objective HPO
                    mo_hpo = False
                    if hasattr(self, "tune_trainer"):
                        if self.tune_trainer.hposearcher.objective.mo_hpo:
                            mo_hpo = True
                    if mo_hpo:
                        invalidOperationError(self.tune_trainer.hposearcher.study,
                                              "You must tune before fit the model.")
                        invalidInputError(use_trial_id is not None,
                                          "For multibojective HPO, "
                                          "you must specify a trial id for fit.")
                        trial = self.tune_trainer.hposearcher.study.trials[use_trial_id]
                        self.internal = self.tune_internal._model_build(trial)

                    # fitting
                    if not validation_data:
                        self.trainer.fit(self.internal, data)
                        self.fitted = True
                    else:
                        if isinstance(validation_data, TSDataset):
                            _rolled = validation_data.numpy_x is None
                            validation_data =\
                                validation_data.to_torch_data_loader(
                                    batch_size=batch_size,
                                    roll=_rolled,
                                    lookback=self.data_config['past_seq_len'],
                                    horizon=self.data_config['future_seq_len'],
                                    feature_col=validation_data.roll_feature,
                                    target_col=validation_data.roll_target,
                                    shuffle=False)
                        if isinstance(validation_data, tuple):
                            validation_data = np_to_dataloader(validation_data, batch_size,
                                                               self.num_processes)
                        self.trainer.fit(self.internal, data, validation_data)
                        self.fitted = True
                        fit_csv = os.path.join(forecaster_log_dir,
                                               "forecaster_tmp_log/version_0/metrics.csv")
                        best_path = os.path.join(validation_ckpt_dir, "best.ckpt")
                        fit_out = read_csv(fit_csv)
                        if validation_mode == 'best_epoch':
                            self.load(best_path)
                        # modify logger attr in trainer, otherwise predict will report error
                        self.trainer._logger_connector.on_trainer_init(
                            False,
                            self.trainer.flush_logs_every_n_steps,
                            self.trainer.log_every_n_steps,
                            self.trainer.move_metrics_to_cpu)
                        return fit_out

    def optimize(self, train_data,
                 validation_data=None,
                 batch_size: int = 32,
                 thread_num: Optional[int] = None,
                 accelerator: Optional[str] = None,
                 precision: Optional[str] = None,
                 metric: str = 'mse',
                 accuracy_criterion: Optional[float] = None):
        '''
        This method will traverse existing optimization methods(onnxruntime, openvino, jit, ...)
        and save the model with minimum latency under the given data and search
        restrictions(accelerator, precision, accuracy_criterion) in `forecaster.accelerated_model`.
        This method is required to call before `predict` and `evaluate`.
        Now this function is only for non-distributed model.

        :param train_data: Data used for training model. Users should be careful with this parameter
               since this data might be exposed to the model, which causing data leak.
               The train_data support following formats:

               | 1. a numpy ndarray tuple (x, y):
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.
               |
               | 2. pytorch dataloader:
               | the dataloader should return x, y in each iteration with the shape as following:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.
               | The batch_size of this dataloader is important as well, users may want to set it
               | to the same batch size you may want to use the model in real deploy environment.
               | E.g. batch size should be set to 1 if you would like to use the accelerated model
               | in an online service.
               |
               | 3. A bigdl.chronos.data.tsdataset.TSDataset instance:
               | Forecaster will automatically process the TSDataset.
               | By default, TSDataset will be transformed to a pytorch dataloader,
               | which is memory-friendly while a little bit slower.
               | Users may call `roll` on the TSDataset before calling `fit`
               | Then the training speed will be faster but will consume more memory.

        :param validation_data(optional): This is only needed when users care about the possible
               accuracy drop. The validation_data support following formats:

               | 1. a numpy ndarray tuple (x, y):
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.
               |
               | 2. pytorch dataloader:
               | the dataloader should return x, y in each iteration with the shape as following:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.
               |
               | 3. A bigdl.chronos.data.tsdataset.TSDataset instance:
               | Forecaster will automatically process the TSDataset.
               | By default, TSDataset will be transformed to a pytorch dataloader,
               | which is memory-friendly while a little bit slower.
               | Users may call `roll` on the TSDataset before calling `fit`
               | Then the training speed will be faster but will consume more memory.

        :param batch_size: Number of batch size you want to use the model in real deploy
               environment. The value defaults to 32. If you input a pytorch dataloader for
               `train_data`, the batch_size will follow the batch_size setted in `train_data`.
        :param thread_num: int, the num of thread limit. The value is set to None by
               default where no limit is set.
        :param accelerator: (optional) Use accelerator 'None', 'onnxruntime',
               'openvino', 'jit', defaults to None. If not None, then will only find the
               model with this specific accelerator.
        :param precision: (optional) Supported type: 'int8', 'bf16', 'fp32'.
               Defaults to None which represents no precision limit. If not None, then will
               only find the model with this specific precision.
        :param metric: (optional) A str represent corresponding metric which is used for calculating
               accuracy.
        :param accuracy_criterion: (optional) a float represents tolerable
               accuracy drop percentage, defaults to None meaning no accuracy control.

        Example:
            >>> # obtain optimized model
            >>> forecaster.optimize(train_data, val_data, thread_num=1)
            >>> pred = forecaster.predict(data)
        '''
        # check distribution
        if self.distributed:
            invalidInputError(False,
                              "optimize has not been supported for distributed "
                              "forecaster. You can call .to_local() to transform the "
                              "forecaster to a non-distributed version.")
        # check fit
        if not self.fitted:
            invalidInputError(False,
                              "You must call fit or restore first before calling optimize!")

        # turn tsdataset to dataloader
        if isinstance(train_data, TSDataset):
            _rolled = train_data.numpy_x is None
            train_data = train_data.to_torch_data_loader(
                batch_size=batch_size,
                roll=_rolled,
                lookback=self.data_config['past_seq_len'],
                horizon=self.data_config['future_seq_len'],
                feature_col=train_data.roll_feature,
                target_col=train_data.roll_target,
                shuffle=False)

        if validation_data is not None and isinstance(validation_data, TSDataset):
            _rolled = validation_data.numpy_x is None
            validation_data = validation_data.to_torch_data_loader(
                batch_size=batch_size,
                roll=_rolled,
                lookback=self.data_config['past_seq_len'],
                horizon=self.data_config['future_seq_len'],
                feature_col=validation_data.roll_feature,
                target_col=validation_data.roll_target,
                shuffle=False)

        # turn numpy into dataloader
        if isinstance(train_data, (tuple, list)):
            invalidInputError(len(train_data) == 2,
                              f"train_data should be a 2-dim tuple, but get {len(train_data)}-dim.")
            train_data = DataLoader(TensorDataset(torch.from_numpy(train_data[0]),
                                                  torch.from_numpy(train_data[1])),
                                    batch_size=batch_size,
                                    shuffle=False)

        if validation_data is not None and isinstance(validation_data, (tuple, list)):
            invalidInputError(len(validation_data) == 2,
                              f"validation_data should be a 2-dim tuple, but get "
                              f"{len(validation_data)}-dim.")
            validation_data = DataLoader(
                TensorDataset(
                    torch.from_numpy(validation_data[0]),
                    torch.from_numpy(validation_data[1])),
                batch_size=batch_size,
                shuffle=False)

        # align metric
        if metric is not None:
            if validation_data is None:
                metric = None
            else:
                try:
                    metric = _str2optimizer_metrc(metric)
                except Exception:
                    invalidInputError(False,
                                      "Unable to recognize the metric string you passed in.")

        dummy_input = torch.rand(1, self.data_config["past_seq_len"],
                                 self.data_config["input_feature_num"])
        # remove channels_last methods and temporarily disable bf16
        excludes = ["fp32_channels_last", "fp32_ipex_channels_last", "bf16_channels_last",
                    "bf16_ipex_channels_last", "jit_fp32_channels_last", "jit_bf16_channels_last",
                    "jit_fp32_ipex_channels_last", "jit_bf16_ipex_channels_last",
                    "bf16", "bf16_ipex", "jit_bf16", "jit_bf16_ipex"]
        if not self.quantize_available:
            excludes = excludes + ["static_int8", "openvino_int8", "onnxruntime_int8_qlinear"]
        from bigdl.chronos.pytorch import TSInferenceOptimizer as InferenceOptimizer
        opt = InferenceOptimizer()
        opt.optimize(model=self.internal,
                     training_data=train_data,
                     validation_data=validation_data,
                     metric=metric,
                     direction="min",
                     thread_num=thread_num,
                     excludes=excludes,
                     input_sample=dummy_input)
        try:
            optim_model, option = opt.get_best_model(
                accelerator=accelerator,
                precision=precision,
                accuracy_criterion=accuracy_criterion)
            self.accelerated_model = optim_model
            self.accelerate_method = option
        except Exception:
            invalidInputError(False, "Unable to find an optimized model that meets your conditions."
                              "Maybe you can relax your search limit.")
        self.optimized_model_thread_num = thread_num

    def predict(self, data, batch_size=32, quantize=False, acceleration: bool = True):
        """
        Predict using a trained forecaster.

        if you want to predict on a single node(which is common practice), please call
        .to_local().predict(x, ...)

        :param data: The data support following formats:

               | 1. a numpy ndarray x:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               |
               | 2. a xshard item:
               | each partition can be a dictionary of {'x': x}, where x's shape
               | should follow the shape stated before.
               |
               | 3. pytorch dataloader:
               | the dataloader needs to return at least x in each iteration
               | with the shape as following:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | If returns x and y only get x.
               |
               | 4. A bigdl.chronos.data.tsdataset.TSDataset instance:
               | Forecaster will automatically process the TSDataset.
               | By default, TSDataset will be transformed to a pytorch dataloader,
               | which is memory-friendly while a little bit slower.
               | Users may call `roll` on the TSDataset before calling `fit`
               | Then the training speed will be faster but will consume memory.

        :param batch_size: predict batch size. The value will not affect predict
               result but will affect resources cost(e.g. memory and time).
        :param quantize: if use the quantized model to predict.
        :param acceleration: bool variable indicates whether use original model.
               Default to True means use accelerated_model to predict, which requires
               to call one of .build_jit(), .build_onnx(), .build_openvino() and
               .optimize(), otherwise the original model will be used to predict.

        :return: A numpy array with shape (num_samples, horizon, target_dim)
                 if data is a numpy ndarray or a dataloader.
                 A xshard item with format {'prediction': result},
                 where result is a numpy array with shape (num_samples, horizon, target_dim)
                 if data is a xshard item.
        """
        from bigdl.chronos.pytorch.utils import _pytorch_fashion_inference
        from bigdl.nano.utils.log4Error import invalidInputError

        if quantize or acceleration:
            self.thread_num = set_pytorch_thread(self.optimized_model_thread_num, self.thread_num)

        if isinstance(data, TSDataset):
            _rolled = data.numpy_x is None
            data = data.to_torch_data_loader(batch_size=batch_size,
                                             roll=_rolled,
                                             lookback=self.data_config['past_seq_len'],
                                             horizon=self.data_config['future_seq_len'],
                                             feature_col=data.roll_feature,
                                             target_col=data.roll_target,
                                             shuffle=False)
        # data transform
        is_local_data = isinstance(data, (np.ndarray, DataLoader))
        if is_local_data and self.distributed:
            if isinstance(data, DataLoader):
                invalidInputError(False,
                                  "We will be support input dataloader later.")
            data = np_to_xshard(data, self.workers_per_node)
        if not is_local_data and not self.distributed:
            data = xshard_to_np(data, mode="predict")

        if self.distributed:
            yhat = self.internal.predict(data, batch_size=batch_size)
            expand_dim = []
            if self.data_config["future_seq_len"] == 1:
                expand_dim.append(1)
            if self.data_config["output_feature_num"] == 1:
                expand_dim.append(2)
            if is_local_data:
                yhat = xshard_to_np(yhat, mode="yhat", expand_dim=expand_dim)
            else:
                yhat = yhat.transform_shard(xshard_expand_dim, expand_dim)
            return yhat
        else:
            if not self.fitted:
                invalidInputError(False,
                                  "You must call fit or restore first before calling predict!")
            if quantize:
                if self.accelerate_method != "pytorch_int8":
                    invalidInputError(False,
                                      "Can't find the quantized model, "
                                      "please call .quantize() method first")
                yhat = _pytorch_fashion_inference(model=self.accelerated_model,
                                                  input_data=data,
                                                  batch_size=batch_size)
            else:
                if acceleration is False or self.accelerated_model is None:
                    self.internal.eval()
                    yhat = _pytorch_fashion_inference(model=self.internal,
                                                      input_data=data,
                                                      batch_size=batch_size)
                else:
                    self.accelerated_model.eval()
                    yhat = _pytorch_fashion_inference(model=self.accelerated_model,
                                                      input_data=data,
                                                      batch_size=batch_size)
            if not is_local_data:
                yhat = np_to_xshard(yhat, self.workers_per_node, prefix="prediction")
            return yhat

    def predict_with_onnx(self, data, batch_size=32, quantize=False):
        """
        Predict using a trained forecaster with onnxruntime. The method can only be
        used when forecaster is a non-distributed version.

        Directly call this method without calling build_onnx is valid and Forecaster will
        automatically build an onnxruntime session with default settings (thread num is 1).

        :param data: The data support following formats:

               | 1. a numpy ndarray x:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               |
               | 2. pytorch dataloader:
               | the dataloader needs to return at least x in each iteration
               | with the shape as following:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | If returns x and y only get x.
               |
               | 3. A bigdl.chronos.data.tsdataset.TSDataset instance:
               | Forecaster will automatically process the TSDataset.
               | By default, TSDataset will be transformed to a pytorch dataloader,
               | which is memory-friendly while a little bit slower.
               | Users may call `roll` on the TSDataset before calling `fit`
               | Then the training speed will be faster but will consume more memory.

        :param batch_size: predict batch size. The value will not affect predict
               result but will affect resources cost(e.g. memory and time). Defaults
               to 32. None for all-data-single-time inference.
        :param quantize: if use the quantized onnx model to predict.

        :return: A numpy array with shape (num_samples, horizon, target_dim).
        """
        from bigdl.chronos.pytorch.utils import _pytorch_fashion_inference
        from bigdl.nano.utils.log4Error import invalidInputError
        if self.distributed:
            invalidInputError(False,
                              "ONNX inference has not been supported for distributed "
                              "forecaster. You can call .to_local() to transform the "
                              "forecaster to a non-distributed version.")
        if not self.fitted:
            invalidInputError(False,
                              "You must call fit or restore first before calling predict!")

        self.thread_num = set_pytorch_thread(self.optimized_model_thread_num, self.thread_num)

        if isinstance(data, TSDataset):
            _rolled = data.numpy_x is None
            data = data.to_torch_data_loader(batch_size=batch_size,
                                             roll=_rolled,
                                             lookback=self.data_config['past_seq_len'],
                                             horizon=self.data_config['future_seq_len'],
                                             feature_col=data.roll_feature,
                                             target_col=data.roll_target,
                                             shuffle=False)
        if quantize:
            if self.accelerate_method != "onnxruntime_int8":
                invalidInputError(False,
                                  "Can't find the quantized model, "
                                  "please call .quantize() method first")
            return _pytorch_fashion_inference(model=self.accelerated_model,
                                              input_data=data,
                                              batch_size=batch_size)
        else:
            if self.accelerate_method != "onnxruntime_fp32":
                self.build_onnx()
                self.thread_num = set_pytorch_thread(self.optimized_model_thread_num,
                                                     self.thread_num)
            return _pytorch_fashion_inference(model=self.accelerated_model,
                                              input_data=data,
                                              batch_size=batch_size)

    def predict_with_openvino(self, data, batch_size=32, quantize=False):
        """
        Predict using a trained forecaster with openvino. The method can only be
        used when forecaster is a non-distributed version.

        Directly call this method without calling build_openvino is valid and Forecaster will
        automatically build an openvino session with default settings (thread num is 1).

       :param data: The data support following formats:

               | 1. a numpy ndarray x:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               |
               | 2. pytorch dataloader:
               | the dataloader needs to return at least x in each iteration
               | with the shape as following:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | If returns x and y only get x.
               |
               | 3. A bigdl.chronos.data.tsdataset.TSDataset instance:
               | Forecaster will automatically process the TSDataset.
               | By default, TSDataset will be transformed to a pytorch dataloader,
               | which is memory-friendly while a little bit slower.
               | Users may call `roll` on the TSDataset before calling `fit`
               | Then the training speed will be faster but will consume more memory.

        :param batch_size: predict batch size. The value will not affect predict
               result but will affect resources cost(e.g. memory and time). Defaults
               to 32. None for all-data-single-time inference.
        :param quantize: if use the quantized openvino model to predict.

        :return: A numpy array with shape (num_samples, horizon, target_dim).
        """
        from bigdl.chronos.pytorch.utils import _pytorch_fashion_inference
        from bigdl.nano.utils.log4Error import invalidInputError

        if self.distributed:
            invalidInputError(False,
                              "Openvino inference has not been supported for distributed "
                              "forecaster. You can call .to_local() to transform the "
                              "forecaster to a non-distributed version.")
        if not self.fitted:
            invalidInputError(False,
                              "You must call fit or restore first before calling predict!")

        self.thread_num = set_pytorch_thread(self.optimized_model_thread_num, self.thread_num)

        if isinstance(data, TSDataset):
            _rolled = data.numpy_x is None
            data = data.to_torch_data_loader(batch_size=batch_size,
                                             roll=_rolled,
                                             lookback=self.data_config['past_seq_len'],
                                             horizon=self.data_config['future_seq_len'],
                                             feature_col=data.roll_feature,
                                             target_col=data.roll_target,
                                             shuffle=False)

        if quantize:
            if self.accelerate_method != "openvino_int8":
                invalidInputError(False,
                                  "Can't find the quantized model, "
                                  "please call .quantize() method first")
            return _pytorch_fashion_inference(model=self.accelerated_model,
                                              input_data=data,
                                              batch_size=batch_size)
        else:
            if self.accelerate_method != "openvino_fp32":
                self.build_openvino()
                self.thread_num = set_pytorch_thread(self.optimized_model_thread_num,
                                                     self.thread_num)
            return _pytorch_fashion_inference(model=self.accelerated_model,
                                              input_data=data,
                                              batch_size=batch_size)

    def predict_with_jit(self, data, batch_size=32, quantize=False):
        """
        Predict using a trained forecaster with jit. The method can only be
        used when forecaster is a non-distributed version.

        Directly call this method without calling build_jit is valid and Forecaster will
        automatically build an jit session with default settings (thread num is 1).

       :param data: The data support following formats:

               | 1. a numpy ndarray x:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               |
               | 2. pytorch dataloader:
               | the dataloader needs to return at least x in each iteration
               | with the shape as following:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | If returns x and y only get x.
               |
               | 3. A bigdl.chronos.data.tsdataset.TSDataset instance:
               | Forecaster will automatically process the TSDataset.
               | By default, TSDataset will be transformed to a pytorch dataloader,
               | which is memory-friendly while a little bit slower.
               | Users may call `roll` on the TSDataset before calling `fit`
               | Then the training speed will be faster but will consume more memory.

        :param batch_size: predict batch size. The value will not affect predict
               result but will affect resources cost(e.g. memory and time). Defaults
               to 32. None for all-data-single-time inference.
        :param quantize: if use the quantized jit model to predict. Not support yet.

        :return: A numpy array with shape (num_samples, horizon, target_dim).
        """
        from bigdl.chronos.pytorch.utils import _pytorch_fashion_inference
        from bigdl.nano.utils.log4Error import invalidInputError

        if self.distributed:
            invalidInputError(False,
                              "Jit inference has not been supported for distributed "
                              "forecaster. You can call .to_local() to transform the "
                              "forecaster to a non-distributed version.")
        if not self.fitted:
            invalidInputError(False,
                              "You must call fit or restore first before calling predict!")

        self.thread_num = set_pytorch_thread(self.optimized_model_thread_num, self.thread_num)

        if isinstance(data, TSDataset):
            _rolled = data.numpy_x is None
            data = data.to_torch_data_loader(batch_size=batch_size,
                                             roll=_rolled,
                                             lookback=self.data_config['past_seq_len'],
                                             horizon=self.data_config['future_seq_len'],
                                             feature_col=data.roll_feature,
                                             target_col=data.roll_target,
                                             shuffle=False)

        if quantize and False:
            if self.accelerate_method != "jit_int8":
                invalidInputError(False,
                                  "Can't find the quantized model, "
                                  "please call .quantize() method first")
            return _pytorch_fashion_inference(model=self.accelerated_model,
                                              input_data=data,
                                              batch_size=batch_size)
        else:
            if self.accelerate_method != "jit_fp32":
                self.build_jit()
                self.thread_num = set_pytorch_thread(self.optimized_model_thread_num,
                                                     self.thread_num)
            return _pytorch_fashion_inference(model=self.accelerated_model,
                                              input_data=data,
                                              batch_size=batch_size)

    def evaluate(self, data, batch_size=32, multioutput="raw_values", quantize=False,
                 acceleration: bool = True):
        """
        Evaluate using a trained forecaster.

        If you want to evaluate on a single node(which is common practice), please call
        .to_local().evaluate(data, ...)

        Please note that evaluate result is calculated by scaled y and yhat. If you scaled
        your data (e.g. use .scale() on the TSDataset), please follow the following code
        snap to evaluate your result if you need to evaluate on unscaled data.

        >>> from bigdl.chronos.metric.forecast_metrics import Evaluator
        >>> y_hat = forecaster.predict(x)
        >>> y_hat_unscaled = tsdata.unscale_numpy(y_hat) # or other customized unscale methods
        >>> y_unscaled = tsdata.unscale_numpy(y) # or other customized unscale methods
        >>> Evaluator.evaluate(metric=..., y_unscaled, y_hat_unscaled, multioutput=...)

        :param data: The data support following formats:

               | 1. a numpy ndarray tuple (x, y):
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.
               |
               | 2. a xshard item:
               | each partition can be a dictionary of {'x': x, 'y': y}, where x and y's shape
               | should follow the shape stated before.
               |
               | 3. pytorch dataloader:
               | the dataloader should return x, y in each iteration with the shape as following:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.
               |
               | 4. A bigdl.chronos.data.tsdataset.TSDataset instance:
               | Forecaster will automatically process the TSDataset.
               | By default, TSDataset will be transformed to a pytorch dataloader,
               | which is memory-friendly while a little bit slower.
               | Users may call `roll` on the TSDataset before calling `fit`
               | Then the training speed will be faster but will consume more memory.

        :param batch_size: evaluate batch size. The value will not affect evaluate
               result but will affect resources cost(e.g. memory and time).
        :param multioutput: Defines aggregating of multiple output values.
               String in ['raw_values', 'uniform_average']. The value defaults to
               'raw_values'.The param is only effective when the forecaster is a
               non-distribtued version.
        :param quantize: if use the quantized model to predict.
        :param acceleration: bool variable indicates whether use original model.
               Default to True means use accelerated_model to predict, which requires
               to call one of .build_jit(), .build_onnx(), .build_openvino() and
               .optimize(), otherwise the original model will be used to predict.

        :return: A list of evaluation results. Each item represents a metric.
        """
        from bigdl.chronos.pytorch.utils import _pytorch_fashion_inference
        from bigdl.nano.utils.log4Error import invalidInputError

        # data transform
        if isinstance(data, TSDataset):
            _rolled = data.numpy_x is None
            data = data.to_torch_data_loader(batch_size=batch_size,
                                             roll=_rolled,
                                             lookback=self.data_config['past_seq_len'],
                                             horizon=self.data_config['future_seq_len'],
                                             feature_col=data.roll_feature,
                                             target_col=data.roll_target,
                                             shuffle=False)
        is_local_data = isinstance(data, (tuple, DataLoader))
        if not is_local_data and not self.distributed:
            data = xshard_to_np(data, mode="fit")
        if self.distributed:
            data = np_to_creator(data) if is_local_data else data
            return self.internal.evaluate(data=data,
                                          batch_size=batch_size)
        else:
            if not self.fitted:
                invalidInputError(False,
                                  "You must call fit or restore first before calling evaluate!")
            if isinstance(data, DataLoader):
                from torch.utils.data.sampler import RandomSampler
                if isinstance(data.sampler, RandomSampler):
                    # If dataloader is shuffled, convert input_data to numpy()
                    # Avoid to iterate shuffled dataloader two times
                    input_x = []
                    input_y = []
                    for val in data:
                        input_x.append(val[0].numpy())
                        input_y.append(val[1].numpy())
                    input_data = np.concatenate(input_x, axis=0)
                    target = np.concatenate(input_y, axis=0)
                else:
                    input_data = data
                    target = np.concatenate(tuple(val[1] for val in data), axis=0)
            else:
                input_data, target = data
            if quantize:
                if self.accelerate_method != "pytorch_int8":
                    invalidInputError(False,
                                      "Can't find the quantized model, "
                                      "please call .quantize() method first")
                yhat = _pytorch_fashion_inference(model=self.accelerated_model,
                                                  input_data=input_data,
                                                  batch_size=batch_size)
            else:
                if acceleration is False or self.accelerated_model is None:
                    self.internal.eval()
                    yhat = _pytorch_fashion_inference(model=self.internal,
                                                      input_data=input_data,
                                                      batch_size=batch_size)
                else:
                    self.accelerated_model.eval()
                    yhat = _pytorch_fashion_inference(model=self.accelerated_model,
                                                      input_data=input_data,
                                                      batch_size=batch_size)

            aggregate = 'mean' if multioutput == 'uniform_average' else None
            return Evaluator.evaluate(self.metrics, target,
                                      yhat, aggregate=aggregate)

    def evaluate_with_onnx(self, data,
                           batch_size=32,
                           multioutput="raw_values",
                           quantize=False):
        """
        Evaluate using a trained forecaster with onnxruntime. The method can only be
        used when forecaster is a non-distributed version.

        Directly call this method without calling build_onnx is valid and Forecaster will
        automatically build an onnxruntime session with default settings (thread num is 1).

        Please note that evaluate result is calculated by scaled y and yhat. If you scaled
        your data (e.g. use .scale() on the TSDataset) please follow the following code
        snap to evaluate your result if you need to evaluate on unscaled data.

        >>> from bigdl.chronos.metric.forecast_metrics import Evaluator
        >>> y_hat = forecaster.predict_with_onnx(x)
        >>> y_hat_unscaled = tsdata.unscale_numpy(y_hat) # or other customized unscale methods
        >>> y_unscaled = tsdata.unscale_numpy(y) # or other customized unscale methods
        >>> Evaluator.evaluate(metric=..., y_unscaled, y_hat_unscaled, multioutput=...)

        :param data: The data support following formats:

               | 1. a numpy ndarray tuple (x, y):
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.
               |
               | 2. pytorch dataloader:
               | should be the same as future_seq_len and output_feature_num.
               | the dataloader should return x, y in each iteration with the shape as following:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               |
               | 3. A bigdl.chronos.data.tsdataset.TSDataset instance:
               | Forecaster will automatically process the TSDataset.
               | By default, TSDataset will be transformed to a pytorch dataloader,
               | which is memory-friendly while a little bit slower.
               | Users may call `roll` on the TSDataset before calling `fit`
               | Then the training speed will be faster but will consume more memory.

        :param batch_size: evaluate batch size. The value will not affect evaluate
               result but will affect resources cost(e.g. memory and time).
        :param multioutput: Defines aggregating of multiple output values.
               String in ['raw_values', 'uniform_average']. The value defaults to
               'raw_values'.
        :param quantize: if use the quantized onnx model to evaluate.

        :return: A list of evaluation results. Each item represents a metric.
        """
        from bigdl.chronos.pytorch.utils import _pytorch_fashion_inference
        from bigdl.nano.utils.log4Error import invalidInputError
        if self.distributed:
            invalidInputError(False,
                              "ONNX inference has not been supported for distributed "
                              "forecaster. You can call .to_local() to transform the "
                              "forecaster to a non-distributed version.")
        if not self.fitted:
            invalidInputError(False,
                              "You must call fit or restore first before calling evaluate!")
        if isinstance(data, TSDataset):
            _rolled = data.numpy_x is None
            data = data.to_torch_data_loader(batch_size=batch_size,
                                             roll=_rolled,
                                             lookback=self.data_config['past_seq_len'],
                                             horizon=self.data_config['future_seq_len'],
                                             feature_col=data.roll_feature,
                                             target_col=data.roll_target,
                                             shuffle=False)
        if isinstance(data, DataLoader):
            from torch.utils.data.sampler import RandomSampler
            if isinstance(data.sampler, RandomSampler):
                # If dataloader is shuffled, convert input_data to numpy()
                # Avoid to iterate shuffled dataloader two times
                input_x = []
                input_y = []
                for val in data:
                    input_x.append(val[0].numpy())
                    input_y.append(val[1].numpy())
                input_data = np.concatenate(input_x, axis=0)
                target = np.concatenate(input_y, axis=0)
            else:
                input_data = data
                target = np.concatenate(tuple(val[1] for val in data), axis=0)
        else:
            input_data, target = data
        if quantize:
            if self.accelerate_method != "onnxruntime_int8":
                invalidInputError(False,
                                  "Can't find the quantized model, "
                                  "please call .quantize() method first")
            yhat = _pytorch_fashion_inference(model=self.accelerated_model,
                                              input_data=input_data,
                                              batch_size=batch_size)
        else:
            if self.accelerate_method != "onnxruntime_fp32":
                self.build_onnx()
            yhat = _pytorch_fashion_inference(model=self.accelerated_model,
                                              input_data=input_data,
                                              batch_size=batch_size)

        aggregate = 'mean' if multioutput == 'uniform_average' else None
        return Evaluator.evaluate(self.metrics, target, yhat, aggregate=aggregate)

    def predict_interval(self, data, validation_data=None, batch_size=32,
                         repetition_times=5):
        """
        Calculate confidence interval of data based on Monte Carlo dropout(MC dropout).
        Related paper : https://arxiv.org/abs/1709.01907

        :param data: The data support following formats:

               | 1. a numpy ndarray x:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               |
               | 2. pytorch dataloader:
               | the dataloader needs to return at least x in each iteration
               | with the shape as following:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | If returns x and y only get x.
               |
               | 3. A bigdl.chronos.data.tsdataset.TSDataset instance:
               | Forecaster will automatically process the TSDataset.
               | By default, TSDataset will be transformed to a pytorch dataloader,
               | which is memory-friendly while a little bit slower.
               | Users may call `roll` on the TSDataset before calling `fit`
               | Then the training speed will be faster but will consume more memory.

        :param validation_data: The validation_data support following formats:

               | 1. a numpy ndarray tuple (x, y):
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.
               |
               | 2. pytorch dataloader:
               | the dataloader should return x, y in each iteration with the shape as following:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.
               |
               | 3. A bigdl.chronos.data.tsdataset.TSDataset instance:
               | Forecaster will automatically process the TSDataset.
               | By default, TSDataset will be transformed to a pytorch dataloader,
               | which is memory-friendly while a little bit slower.
               | Users may call `roll` on the TSDataset before calling `fit`
               | Then the training speed will be faster but will consume more memory.

        :param batch_size: predict batch size. The value will not affect predict
               result but will affect resources cost(e.g. memory and time).
        :param repetition_times: Defines repeate how many times to calculate model
                                 uncertainty based on MC Dropout.

        :return: prediction and standard deviation which are both numpy array
                 with shape (num_samples, horizon, target_dim)

        """
        from bigdl.chronos.pytorch.utils import _pytorch_fashion_inference

        if self.distributed:
            invalidInputError(False,
                              "predict interval has not been supported for distributed "
                              "forecaster. You can call .to_local() to transform the "
                              "forecaster to a non-distributed version.")

        if not self.fitted:
            invalidInputError(False,
                              "You must call fit or restore first before calling predict_interval!")

        self.thread_num = set_pytorch_thread(self.optimized_model_thread_num, self.thread_num)

        # step1, according to validation dataset, calculate inherent noise
        if not hasattr(self, "data_noise"):
            invalidInputError(validation_data is not None,
                              "When call predict_interval for the first time, you must pass in "
                              "validation_data to calculate data noise.")
            # data transform
            if isinstance(validation_data, TSDataset):
                _rolled = validation_data.numpy_x is None
                validation_data = validation_data.to_torch_data_loader(
                    batch_size=batch_size,
                    roll=_rolled,
                    lookback=self.data_config['past_seq_len'],
                    horizon=self.data_config['future_seq_len'],
                    feature_col=data.roll_feature,
                    target_col=data.roll_target,
                    shuffle=False,
                )

            if isinstance(validation_data, DataLoader):
                input_data = validation_data
                target = np.concatenate(tuple(val[1] for val in validation_data), axis=0)
            else:
                input_data, target = validation_data
            self.internal.eval()
            val_yhat = _pytorch_fashion_inference(model=self.internal,
                                                  input_data=input_data,
                                                  batch_size=batch_size)
            self.data_noise = Evaluator.evaluate(["mse"], target,
                                                 val_yhat, aggregate=None)[0]  # 2d array

        # data preprocess
        if isinstance(data, TSDataset):
            _rolled = data.numpy_x is None
            data = data.to_torch_data_loader(batch_size=batch_size,
                                             roll=_rolled,
                                             lookback=self.data_config['past_seq_len'],
                                             horizon=self.data_config['future_seq_len'],
                                             feature_col=data.roll_feature,
                                             target_col=data.roll_target,
                                             shuffle=False)

        # step2: calculate model uncertainty based MC Dropout
        def apply_dropout(m):
            if type(m) == torch.nn.Dropout:
                m.train()

        # turn on dropout
        self.internal.apply(apply_dropout)

        y_hat_list = []
        for i in range(repetition_times):
            _yhat = _pytorch_fashion_inference(model=self.internal,
                                               input_data=data,
                                               batch_size=batch_size)
            y_hat_list.append(_yhat)
        y_hat_mean = np.mean(np.stack(y_hat_list, axis=0), axis=0)

        model_bias = np.zeros_like(y_hat_mean)  # 3d array
        for i in range(repetition_times):
            model_bias += (y_hat_list[i] - y_hat_mean)**2
        model_bias /= repetition_times
        std_deviation = np.sqrt(self.data_noise + model_bias)

        return y_hat_mean, std_deviation

    def save(self, checkpoint_file, quantize_checkpoint_file=None):
        """
        Save the forecaster.

        Please note that if you only want the pytorch model or onnx model
        file, you can call .get_model() or .export_onnx_file(). The checkpoint
        file generated by .save() method can only be used by .load().

        :param checkpoint_file: The location you want to save the forecaster.
        :param quantize_checkpoint_file: The location you want to save quantized forecaster.
        """
        from bigdl.chronos.pytorch import TSTrainer as Trainer

        if self.distributed:
            self.internal.save(checkpoint_file)
        else:
            if not self.fitted:
                from bigdl.nano.utils.log4Error import invalidInputError
                invalidInputError(False,
                                  "You must call fit or restore first before calling save!")
            # user may never call the fit before
            if self.trainer.model is None:
                self.trainer.model = self.internal
            self.trainer.save_checkpoint(checkpoint_file)  # save current status
            if quantize_checkpoint_file:
                if self.accelerate_method == "pytorch_int8":
                    Trainer.save(self.accelerated_model, quantize_checkpoint_file)
                else:
                    warnings.warn("Please call .quantize() method to build "
                                  "an up-to-date quantized model")

    def load(self, checkpoint_file, quantize_checkpoint_file=None):
        """
        restore the forecaster.

        :param checkpoint_file: The checkpoint file location you want to load the forecaster.
        :param quantize_checkpoint_file: The checkpoint file location you want to
               load the quantized forecaster.
        """
        from bigdl.chronos.pytorch import TSTrainer as Trainer

        if self.distributed:
            self.internal.load(checkpoint_file)
        else:
            from bigdl.nano.pytorch.lightning import LightningModule
            from bigdl.chronos.pytorch import TSTrainer as Trainer
            if self.use_hpo:
                ckpt = torch.load(checkpoint_file)
                hparams = ckpt["hyper_parameters"]
                model = self.model_creator(hparams)
                loss = self.loss_creator(hparams)
                optimizer = self.optimizer_creator(model, hparams)
            else:
                model = self.model_creator({**self.model_config, **self.data_config})
                loss = self.loss_creator(self.loss_config)
                optimizer = self.optimizer_creator(model, self.optim_config)
            self.internal = LightningModule.load_from_checkpoint(checkpoint_file,
                                                                 model=model,
                                                                 loss=loss,
                                                                 optimizer=optimizer)
            self.internal = Trainer.compile(self.internal)
            self.fitted = True
            if quantize_checkpoint_file:
                # self.internal.load_quantized_state_dict(torch.load(quantize_checkpoint_file))
                self.accelerated_model = Trainer.load(quantize_checkpoint_file,
                                                      self.internal)
                self.accelerate_method = "pytorch_int8"
            # This trainer is only for quantization, once the user call `fit`, it will be
            # replaced according to the new training config
            self.trainer = Trainer(logger=False, max_epochs=1,
                                   enable_checkpointing=self.checkpoint_callback,
                                   num_processes=self.num_processes, use_ipex=self.use_ipex)

    def to_local(self):
        """
        Transform a distributed forecaster to a local (non-distributed) one.

        Common practice is to use distributed training (fit) and predict/
        evaluate with onnx or other frameworks on a single node. To do so,
        you need to call .to_local() and transform the forecaster to a non-
        distributed one.

        The optimizer is refreshed, incremental training after to_local
        might have some problem.

        :return: a forecaster instance.
        """
        from bigdl.chronos.pytorch import TSTrainer as Trainer
        from bigdl.nano.utils.log4Error import invalidInputError
        # TODO: optimizer is refreshed, which is not reasonable
        if not self.distributed:
            invalidInputError(False, "The forecaster has become local.")
        model = self.internal.get_model()
        self.internal.shutdown()

        loss = self.loss_creator(self.loss_config)
        optimizer = self.optimizer_creator(model, self.optim_config)
        self.internal = Trainer.compile(model=model, loss=loss,
                                        optimizer=optimizer)
        # This trainer is only for saving, once the user call `fit`, it will be
        # replaced according to the new training config
        self.trainer = Trainer(logger=False, max_epochs=1,
                               enable_checkpointing=self.checkpoint_callback,
                               num_processes=self.num_processes, use_ipex=self.use_ipex)

        self.distributed = False
        self.fitted = True

        # placeholder for accelerated model obtained from various accelerators
        self.accelerated_model = None
        # str indicates current accelerate method
        self.accelerate_method = None
        return self

    def get_model(self):
        """
        Returns the learned PyTorch model.

        :return: a pytorch model instance
        """
        if self.distributed:
            return self.internal.get_model()
        else:
            return self.internal.model

    def build_onnx(self, thread_num=1, sess_options=None):
        '''
        Build onnx model to speed up inference and reduce latency.
        The method is Not required to call before predict_with_onnx,
        evaluate_with_onnx or export_onnx_file.
        It is recommended to use when you want to:

        | 1. Strictly control the thread to be used during inferencing.
        | 2. Alleviate the cold start problem when you call predict_with_onnx
             for the first time.

        :param thread_num: int, the num of thread limit. The value is set to None by
               default where no limit is set. Besides, the environment variable
               `OMP_NUM_THREADS` is suggested to be same as `thread_num`.
        :param sess_options: an onnxruntime.SessionOptions instance, if you set this
               other than None, a new onnxruntime session will be built on this setting
               and ignore other settings you assigned(e.g. thread_num...).

        Example:
            >>> # to pre build onnx sess
            >>> forecaster.build_onnx(thread_num=2)  # build onnx runtime sess for two threads
            >>> pred = forecaster.predict_with_onnx(data)
            >>> # ------------------------------------------------------
            >>> # directly call onnx related method is also supported
            >>> # default to build onnx runtime sess for single thread
            >>> pred = forecaster.predict_with_onnx(data)
        '''
        import onnxruntime
        from bigdl.chronos.pytorch import TSInferenceOptimizer as InferenceOptimizer
        from bigdl.nano.utils.log4Error import invalidInputError
        if sess_options is not None and not isinstance(sess_options, onnxruntime.SessionOptions):
            invalidInputError(False,
                              "sess_options should be an onnxruntime.SessionOptions instance"
                              f", but found {type(sess_options)}")
        if sess_options is None:
            sess_options = onnxruntime.SessionOptions()
            if thread_num is not None:
                sess_options.intra_op_num_threads = thread_num
                sess_options.inter_op_num_threads = thread_num
        if self.distributed:
            invalidInputError(False,
                              "build_onnx has not been supported for distributed "
                              "forecaster. You can call .to_local() to transform the "
                              "forecaster to a non-distributed version.")
        try:
            OMP_NUM_THREADS = os.getenv("OMP_NUM_THREADS")
        except KeyError:
            OMP_NUM_THREADS = 0
        if OMP_NUM_THREADS != str(thread_num):
            warnings.warn("The environment variable OMP_NUM_THREADS is suggested to be same "
                          f"as thread_num.You can use 'export OMP_NUM_THREADS={thread_num}'.")
        dummy_input = torch.rand(1, self.data_config["past_seq_len"],
                                 self.data_config["input_feature_num"])
        self.accelerated_model = InferenceOptimizer.trace(self.internal,
                                                          input_sample=dummy_input,
                                                          accelerator="onnxruntime",
                                                          onnxruntime_session_options=sess_options)
        self.accelerate_method = "onnxruntime_fp32"
        self.optimized_model_thread_num = thread_num

    def build_openvino(self, thread_num=1):
        '''
        Build openvino model to speed up inference and reduce latency.
        The method is Not required to call before predict_with_openvino.

        It is recommended to use when you want to:

        | 1. Strictly control the thread to be used during inferencing.
        | 2. Alleviate the cold start problem when you call predict_with_openvino
             for the first time.

        :param thread_num: int, the num of thread limit. The value is set to 1 by
               default where no limit is set. Besides, the environment variable
               `OMP_NUM_THREADS` is suggested to be same as `thread_num`.
        '''
        from bigdl.chronos.pytorch import TSInferenceOptimizer as InferenceOptimizer
        from bigdl.nano.utils.log4Error import invalidInputError

        if self.distributed:
            invalidInputError(False,
                              "build_openvino has not been supported for distributed "
                              "forecaster. You can call .to_local() to transform the "
                              "forecaster to a non-distributed version.")
        try:
            OMP_NUM_THREADS = os.getenv("OMP_NUM_THREADS")
        except KeyError:
            OMP_NUM_THREADS = 0
        if OMP_NUM_THREADS != str(thread_num):
            warnings.warn("The environment variable OMP_NUM_THREADS is suggested to be same "
                          f"as thread_num.You can use 'export OMP_NUM_THREADS={thread_num}'.")
        dummy_input = torch.rand(1, self.data_config["past_seq_len"],
                                 self.data_config["input_feature_num"])
        self.accelerated_model = InferenceOptimizer.trace(self.internal,
                                                          input_sample=dummy_input,
                                                          accelerator="openvino",
                                                          thread_num=thread_num)
        self.accelerate_method = "openvino_fp32"
        self.optimized_model_thread_num = thread_num

    def build_jit(self, thread_num=1, use_ipex=False):
        '''
         Build jit model to speed up inference and reduce latency.
         The method is Not required to call before predict_with_jit
         or export_torchscript_file.

         It is recommended to use when you want to:

         | 1. Strictly control the thread to be used during inferencing.
         | 2. Alleviate the cold start problem when you call predict_with_jit
         |    for the first time.

         :param use_ipex: if to use intel-pytorch-extension for acceleration. Typically,
                intel-pytorch-extension will bring some acceleration while causing some
                unexpected error as well.
         :param thread_num: int, the num of thread limit. The value is set to 1 by
                default where no limit is set.Besides, the environment variable
               `OMP_NUM_THREADS` is suggested to be same as `thread_num`.
         '''
        from bigdl.nano.pytorch import InferenceOptimizer
        from bigdl.nano.utils.log4Error import invalidInputError

        if self.distributed:
            invalidInputError(False,
                              "build_jit has not been supported for distributed "
                              "forecaster. You can call .to_local() to transform the "
                              "forecaster to a non-distributed version.")
        try:
            OMP_NUM_THREADS = os.getenv("OMP_NUM_THREADS")
        except KeyError:
            OMP_NUM_THREADS = 0
        if OMP_NUM_THREADS != str(thread_num):
            warnings.warn("The environment variable OMP_NUM_THREADS is suggested to be same "
                          f"as thread_num.You can use 'export OMP_NUM_THREADS={thread_num}'.")
        dummy_input = torch.rand(1, self.data_config["past_seq_len"],
                                 self.data_config["input_feature_num"])
        self.accelerated_model = InferenceOptimizer.trace(self.internal,
                                                          input_sample=dummy_input,
                                                          accelerator="jit",
                                                          use_ipex=use_ipex,
                                                          channels_last=False,
                                                          thread_num=thread_num)
        self.accelerate_method = "jit_fp32"
        self.optimized_model_thread_num = thread_num

    def export_onnx_file(self, dirname="fp32_onnx", quantized_dirname=None):
        """
        Save the onnx model file to the disk.

        :param dirname: The dir location you want to save the onnx file.
        :param quantized_dirname: The dir location you want to save the quantized onnx file.
        """
        from bigdl.chronos.pytorch import TSInferenceOptimizer as InferenceOptimizer
        from bigdl.nano.utils.log4Error import invalidInputError
        if self.distributed:
            invalidInputError(False,
                              "export_onnx_file has not been supported for distributed "
                              "forecaster. You can call .to_local() to transform the "
                              "forecaster to a non-distributed version.")
        if quantized_dirname:
            if self.accelerate_method == "onnxruntime_int8":
                InferenceOptimizer.save(self.accelerated_model, quantized_dirname)
            else:
                warnings.warn("Please call .quantize() method to build "
                              "an up-to-date quantized model")
        if dirname:
            if self.accelerate_method != "onnxruntime_fp32":
                self.build_onnx()
            InferenceOptimizer.save(self.accelerated_model, dirname)

    def export_openvino_file(self, dirname="fp32_openvino",
                             quantized_dirname=None):
        """
        Save the openvino model file to the disk.

        :param dirname: The dir location you want to save the openvino file.
        :param quantized_dirname: The dir location you want to save the quantized openvino file.
        """
        from bigdl.chronos.pytorch import TSInferenceOptimizer as InferenceOptimizer
        from bigdl.nano.utils.log4Error import invalidInputError
        if self.distributed:
            invalidInputError(False,
                              "export_openvino_file has not been supported for distributed "
                              "forecaster. You can call .to_local() to transform the "
                              "forecaster to a non-distributed version.")
        if quantized_dirname:
            if self.accelerate_method == "openvino_int8":
                InferenceOptimizer.save(self.accelerated_model, quantized_dirname)
            else:
                warnings.warn("Please call .quantize() method to build "
                              "an up-to-date quantized model")
        if dirname:
            if self.accelerate_method != "openvino_fp32":
                self.build_openvino()
            InferenceOptimizer.save(self.accelerated_model, dirname)

    def export_torchscript_file(self, dirname="fp32_torchscript",
                                quantized_dirname=None,
                                save_pipeline=False,
                                tsdata=None,
                                drop_dt_col=True):
        """
        Save the torchscript model file and the whole forecasting pipeline to the disk.

        When the whole forecasting pipeline is saved, it can be used without Python environment.
        For example, when you finish developing a forecaster, you could call this method with
        "save_pipeline=True" to save the whole pipeline (data preprocessing, inference, data
        postprocessing) to torchscript (the forecaster will be saved as torchscript model too),
        then you could deploy the pipeline in C++ using libtorch APIs.

        Currently the pipeline is similar to the following code:

        >>> # preprocess
        >>> tsdata.scale(scaler, fit=False) \\
        >>>       .roll(lookback, horizon, is_predict=True)
        >>> preprocess_output = tsdata.to_numpy()
        >>> # inference using trained forecaster
        >>> # forecaster_module is the saved torchscript model
        >>> inference_output = forecaster_module.forward(preprocess_output)
        >>> # postprocess
        >>> postprocess_output = tsdata.unscale_numpy(inference_output)

        When deploying, the pipeline can be used by:

        >>> // deployment in C++
        >>> #include <torch/torch.h>
        >>> #include <torch/script.h>
        >>> // create input tensor from your data
        >>> // The data to create the input tensor should have the same format as the
        >>> // data used in developing
        >>> torch::Tensor input = create_input_tensor(data);
        >>> // load the pipeline
        >>> torch::jit::script::Module forecasting_pipeline;
        >>> forecasting_pipeline = torch::jit::load(path);
        >>> // run pipeline
        >>> torch::Tensor output = forecasting_pipeline.forward(input_tensor).toTensor();

        The limitations of exporting the forecasting pipeline is same as limitations in
        TSDataset.export_jit():
            1. Please make sure the value of each column can be converted to Pytorch tensor,
               for example, id "00" is not allowed because str can not be converted to a tensor,
               you should use integer (0, 1, ..) as id instead of string.
            2. Some features in tsdataset.scale and tsdataset.roll are unavailable in this
               pipeline:
                    a. If self.roll_additional_feature is not None, it can't be processed in scale
                       and roll
                    b. id_sensitive, time_enc and label_len parameter is not supported in roll
            3. Users are expected to call .scale(scaler, fit=True) before calling export_jit.
               Single roll operation is not supported for converting now.

        :param dirname: The dir location you want to save the torchscript file.
        :param quantized_dirname: The dir location you want to save the quantized torchscript model.
        :param save_pipeline: Whether to save the whole forecasting pipeline, defaluts to False.
               If set to True, the whole forecasting pipeline will be saved in
               "dirname/chronos_forecasting_pipeline.pt", if set to False, only the torchscript
               model will be saved.
        :param tsdata: The TSDataset instance used when developing the forecaster. The parameter
               should be used only when save_pipeline is True.
        :param drop_dt_col: Whether to delete the datetime column, defaults to True. The parameter
               is valid only when save_pipeline is True.
               Since datetime value (like "2022-12-12") can't be converted to Pytorch tensor, you
               can choose different ways to workaround this. If set to True, the datetime column
               will be deleted, then you also need to skip the datetime column when reading data
               from data source (like csv files) in deployment environment to keep the same
               structure as the data used in development; if set to False, the datetime column will
               not be deleted, and you need to make sure the datetime colunm can be successfully
               converted to Pytorch tensor when reading data in deployment environment. For
               example, you can set each data in datetime column to an int (or other vaild types)
               value, since datetime column is not necessary in preprocessing and postprocessing,
               the value can be arbitrary.
        """
        from bigdl.nano.pytorch import InferenceOptimizer
        from bigdl.nano.utils.log4Error import invalidInputError
        from pathlib import Path
        if self.distributed:
            invalidInputError(False,
                              "export_torchscript_file has not been supported for distributed "
                              "forecaster. You can call .to_local() to transform the "
                              "forecaster to a non-distributed version.")
        if quantized_dirname:
            if self.accelerate_method == "jit_int8":
                InferenceOptimizer.save(self.accelerated_model, quantized_dirname)
            else:
                warnings.warn("Please call .quantize() method to build "
                              "an up-to-date quantized model")
        if dirname:
            if self.accelerate_method != "jit_fp32":
                self.build_jit()
            InferenceOptimizer.save(self.accelerated_model, dirname)

            if save_pipeline:
                forecaster_path = Path(dirname) / "ckpt.pth"
                exproted_module = get_exported_module(tsdata, forecaster_path, drop_dt_col)
                saved_path = Path(dirname) / "chronos_forecasting_pipeline.pt"
                torch.jit.save(exproted_module, saved_path)

    def quantize(self, calib_data=None,
                 val_data=None,
                 metric=None,
                 conf=None,
                 framework='pytorch_fx',
                 approach='static',
                 tuning_strategy='bayesian',
                 relative_drop=None,
                 absolute_drop=None,
                 timeout=0,
                 max_trials=1,
                 sess_options=None,
                 thread_num=None):
        """
        Quantize the forecaster.

        :param calib_data: Required for static quantization. Support following formats:

               | 1. a numpy ndarray tuple (x, y):
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.
               |
               | 2. pytorch dataloader:
               | the dataloader should return x, y in each iteration with the shape as following:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.
               |
               | 3. A bigdl.chronos.data.tsdataset.TSDataset instance:
               | Forecaster will automatically process the TSDataset.
               | By default, TSDataset will be transformed to a pytorch dataloader,
               | which is memory-friendly while a little bit slower.
               | Users may call `roll` on the TSDataset before calling `fit`
               | Then the training speed will be faster but will consume more memory.

        :param val_data: for evaluation. Support following formats:

               | 1. a numpy ndarray tuple (x, y):
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.
               |
               | 2. pytorch dataloader:
               | the dataloader should return x, y in each iteration with the shape as following:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.
               |
               | 3. A bigdl.chronos.data.tsdataset.TSDataset instance:
               | Forecaster will automatically process the TSDataset.
               | By default, TSDataset will be transformed to a pytorch dataloader,
               | which is memory-friendly while a little bit slower.
               | Users may call `roll` on the TSDataset before calling `fit`
               | Then the training speed will be faster but will consume more memory.

        :param metric: A str represent the metrics for tunning the quality of
               quantization. You may choose from "mse", "mae", "rmse", "r2", "mape", "smape".
        :param conf: A path to conf yaml file for quantization. Default to None,
               using default config.
        :param framework: A str represent the framework for quantization. You may choose from
               "pytorch_fx", "pytorch_ipex", "onnxrt_integerops", "onnxrt_qlinearops", "openvino".
               Default: 'pytorch_fx'. Consistent with Intel Neural Compressor.
        :param approach: str, 'static' or 'dynamic'. Default to 'static'.
               OpenVINO supports static mode only, if set to 'dynamic',
               it will be replaced with 'static'.
        :param tuning_strategy: str, 'bayesian', 'basic', 'mse' or 'sigopt'. Default to 'bayesian'.
        :param relative_drop: Float, tolerable ralative accuracy drop. Default to None,
               e.g. set to 0.1 means that we accept a 10% increase in the metrics error.
        :param absolute_drop: Float, tolerable ralative accuracy drop. Default to None,
               e.g. set to 5 means that we can only accept metrics smaller than 5.
        :param timeout: Tuning timeout (seconds). Default to 0, which means early stop.
               Combine with max_trials field to decide when to exit.
        :param max_trials: Max tune times. Default to 1. Combine with timeout field to
               decide when to exit. "timeout=0, max_trials=1" means it will try quantization
               only once and return satisfying best model.
        :param sess_options: The session option for onnxruntime, only valid when
                             framework contains 'onnxrt_integerops' or 'onnxrt_qlinearops',
                             otherwise will be ignored.
        :param thread_num: int, the num of thread limit. The value is set to None by
               default where no limit is set
        """
        # check model support for quantization
        from bigdl.nano.utils.log4Error import invalidInputError
        from bigdl.chronos.pytorch import TSInferenceOptimizer as InferenceOptimizer
        if not self.quantize_available:
            invalidInputError(False,
                              "This model has not supported quantization.")

        # Distributed forecaster does not support quantization
        if self.distributed:
            invalidInputError(False,
                              "quantization has not been supported for distributed "
                              "forecaster. You can call .to_local() to transform the "
                              "forecaster to a non-distributed version.")

        # calib data should be set correctly according to the approach
        if approach == 'static' and calib_data is None:
            invalidInputError(False, "You must set a `calib_data` for static quantization.")
        if approach == 'dynamic' and calib_data is not None:
            invalidInputError(False, "You must not set a `calib_data` for dynamic quantization.")

        # change tsdataset to dataloader
        if isinstance(calib_data, TSDataset):
            _rolled = calib_data.numpy_x is None
            calib_data = calib_data.to_torch_data_loader(
                batch_size=1,
                roll=_rolled,
                lookback=self.data_config['past_seq_len'],
                horizon=self.data_config['future_seq_len'],
                feature_col=calib_data.roll_feature,
                target_col=calib_data.roll_target,
                shuffle=False)
        if isinstance(val_data, TSDataset):
            _rolled = val_data.numpy_x is None
            val_data = val_data.to_torch_data_loader(
                batch_size=1,
                roll=_rolled,
                lookback=self.data_config['past_seq_len'],
                horizon=self.data_config['future_seq_len'],
                feature_col=val_data.roll_feature,
                target_col=val_data.roll_target,
                shuffle=False)

        # change data tuple to dataloader
        if isinstance(calib_data, tuple):
            calib_data = DataLoader(TensorDataset(torch.from_numpy(calib_data[0]),
                                                  torch.from_numpy(calib_data[1])))
        if isinstance(val_data, tuple):
            val_data = DataLoader(TensorDataset(torch.from_numpy(val_data[0]),
                                                torch.from_numpy(val_data[1])))

        metric = _str2metric(metric)

        # init acc criterion
        accuracy_criterion = None
        if relative_drop and absolute_drop:
            invalidInputError(False, "Please unset either `relative_drop` or `absolute_drop`.")
        if relative_drop:
            accuracy_criterion = {'relative': relative_drop, 'higher_is_better': False}
        if absolute_drop:
            accuracy_criterion = {'absolute': absolute_drop, 'higher_is_better': False}

        # quantize
        if '_' in framework:
            accelerator, method = framework.split('_')
        else:
            accelerator = framework
        if accelerator == 'pytorch':
            accelerator = None
        elif accelerator == 'openvino':
            method = None
            approach = "static"
        else:
            accelerator = 'onnxruntime'
            method = method[:-3]
        q_model = InferenceOptimizer.quantize(self.internal,
                                              precision='int8',
                                              accelerator=accelerator,
                                              method=method,
                                              calib_data=calib_data,
                                              metric=metric,
                                              conf=conf,
                                              approach=approach,
                                              tuning_strategy=tuning_strategy,
                                              accuracy_criterion=accuracy_criterion,
                                              timeout=timeout,
                                              max_trials=max_trials,
                                              onnxruntime_session_options=sess_options,
                                              thread_num=thread_num)
        if accelerator == 'onnxruntime':
            self.accelerated_model = q_model
            self.accelerate_method = "onnxruntime_int8"
        if accelerator == 'openvino':
            self.accelerated_model = q_model
            self.accelerate_method = "openvino_int8"
        if accelerator is None:
            self.accelerated_model = q_model
            self.accelerate_method = "pytorch_int8"
        self.optimized_model_thread_num = thread_num

    @classmethod
    def from_tsdataset(cls, tsdataset, past_seq_len=None, future_seq_len=None, **kwargs):
        """
        Build a Forecaster Model.

        :param tsdataset: Train tsdataset, a bigdl.chronos.data.tsdataset.TSDataset instance.
        :param past_seq_len: int or "auto", Specify the history time steps (i.e. lookback).
               Do not specify the 'past_seq_len' if your tsdataset has called
               the 'TSDataset.roll' method or 'TSDataset.to_torch_data_loader'.
               If "auto", the mode of time series' cycle length will be taken as the past_seq_len.
        :param future_seq_len: int or list, Specify the output time steps (i.e. horizon).
               Do not specify the 'future_seq_len' if your tsdataset has called
               the 'TSDataset.roll' method or 'TSDataset.to_torch_data_loader'.
        :param kwargs: Specify parameters of Forecaster,
               e.g. loss and optimizer, etc.
               More info, please refer to Forecaster.__init__ methods.

        :return: A Forecaster Model.
        """
        from bigdl.nano.utils.log4Error import invalidInputError
        invalidInputError(isinstance(tsdataset, TSDataset),
                          f"We only supports input a TSDataset, but get{type(tsdataset)}.")

        def check_time_steps(tsdataset, past_seq_len, future_seq_len):
            if tsdataset.lookback is not None and past_seq_len is not None:
                future_seq_len = future_seq_len if isinstance(future_seq_len, int)\
                    else max(future_seq_len)
                return tsdataset.lookback == past_seq_len and tsdataset.horizon == future_seq_len
            return True

        invalidInputError(not tsdataset._has_generate_agg_feature,
                          "We will add support for 'gen_rolling_feature' method later.")

        if tsdataset.lookback is not None:  # calling roll or to_torch_data_loader
            past_seq_len = tsdataset.lookback
            future_seq_len = tsdataset.horizon if isinstance(tsdataset.horizon, int) \
                else max(tsdataset.horizon)
            output_feature_num = len(tsdataset.roll_target)
            input_feature_num = len(tsdataset.roll_feature) + output_feature_num
        elif past_seq_len is not None and future_seq_len is not None:  # initialize only
            past_seq_len = past_seq_len if isinstance(past_seq_len, int)\
                else tsdataset.get_cycle_length()
            future_seq_len = future_seq_len if isinstance(future_seq_len, int) \
                else max(future_seq_len)
            output_feature_num = len(tsdataset.target_col)
            input_feature_num = len(tsdataset.feature_col) + output_feature_num
        else:
            invalidInputError(False,
                              "Forecaster requires 'past_seq_len' and 'future_seq_len' to specify "
                              "the history time step and output time step.")

        invalidInputError(check_time_steps(tsdataset, past_seq_len, future_seq_len),
                          "tsdataset already has history time steps and "
                          "differs from the given past_seq_len and future_seq_len "
                          "Expected past_seq_len and future_seq_len to be "
                          f"{tsdataset.lookback, tsdataset.horizon}, "
                          f"but found {past_seq_len, future_seq_len}.",
                          fixMsg="Do not specify past_seq_len and future seq_len "
                          "or call tsdataset.roll method again and specify time step")

        if tsdataset.id_sensitive:
            _id_list_len = len(tsdataset.id_col)
            input_feature_num *= _id_list_len
            output_feature_num *= _id_list_len

        return cls(past_seq_len=past_seq_len,
                   future_seq_len=future_seq_len,
                   input_feature_num=input_feature_num,
                   output_feature_num=output_feature_num,
                   **kwargs)


def _str2metric(metric):
    # map metric str to function
    if isinstance(metric, str):
        metric_name = metric
        from bigdl.chronos.metric.forecast_metrics import REGRESSION_MAP
        metric_func = REGRESSION_MAP[metric_name]

        def metric(y_label, y_predict):
            y_label = y_label.numpy()
            y_predict = y_predict.numpy()
            return metric_func(y_label, y_predict)
        metric.__name__ = metric_name
    return metric


def _str2optimizer_metrc(metric):
    # map metric str to function for InferenceOptimizer
    if isinstance(metric, str):
        metric_name = metric
        from bigdl.chronos.metric.forecast_metrics import REGRESSION_MAP
        metric_func = REGRESSION_MAP[metric_name]

        def metric(pred, target):
            pred = pred.numpy()
            target = target.numpy()
            return metric_func(target, pred)
        metric.__name__ = metric_name
    return metric
