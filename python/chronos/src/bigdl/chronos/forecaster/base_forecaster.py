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

import numpy as np
import warnings
import torch
import math
from functools import partial
from torch.utils.data import TensorDataset, DataLoader
from bigdl.nano.automl.hpo.space import Space
from .utils_hpo import GenericLightningModule, _format_metric_str, _config_has_search_space
from bigdl.nano.utils.log4Error import invalidOperationError, invalidInputError


class BasePytorchForecaster(Forecaster):
    '''
    Forecaster base model for lstm, seq2seq, tcn and nbeats forecasters.
    '''
    def __init__(self, **kwargs):
        self.internal = None
        if self.distributed:
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
                                                 backend=self.distributed_backend,
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
                if self.use_hpo:
                    warnings.warn("HPO is enabled but no spaces is specified, so disable HPO.")
                self.use_hpo = False
                model = self.model_creator({**self.model_config, **self.data_config})
                loss = self.loss_creator(self.loss_config)
                optimizer = self.optimizer_creator(model, self.optim_config)
                self.internal = Trainer.compile(model=model, loss=loss,
                                                optimizer=optimizer)
            self.onnxruntime_fp32 = None  # onnxruntime session for fp32 precision
            self.openvino_fp32 = None  # placeholader openvino session for fp32 precision
            self.onnxruntime_int8 = None  # onnxruntime session for int8 precision
            self.pytorch_int8 = None  # pytorch model for int8 precision

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
             n_trials=2,
             n_parallels=1,
             epochs=1,
             batch_size=32,
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

        # prepare target metric
        if validation_data is not None:
            formated_target_metric = _format_metric_str('val', target_metric)
        else:
            invalidInputError(False, "To use tuning, you must provide validation_data"
                                     "as numpy arrays.")

        # build auto model
        self.tune_internal = self._build_automodel(data, validation_data, batch_size, epochs)

        from pytorch_lightning.callbacks import Callback

        # reset current epoch = 0 after each run
        class ResetCallback(Callback):
            def on_train_end(self, trainer, pl_module):
                trainer.fit_loop.current_epoch = 0

        # shall we use the same trainier
        self.tune_trainer = Trainer(logger=False, max_epochs=epochs,
                                    checkpoint_callback=self.checkpoint_callback,
                                    num_processes=self.num_processes, use_ipex=self.use_ipex,
                                    use_hpo=True,
                                    callbacks=[ResetCallback()] if self.num_processes == 1
                                    else None)
        # run hyper parameter search
        self.internal = self.tune_trainer.search(
            self.tune_internal,
            n_trials=n_trials,
            target_metric=formated_target_metric,
            direction=direction,
            n_parallels=n_parallels,
            **kwargs)

        # reset train and validation datasets
        self.tune_trainer.reset_train_val_dataloaders(self.internal)

    def fit(self, data, epochs=1, batch_size=32):
        # TODO: give an option to close validation during fit to save time.
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

        :param epochs: Number of epochs you want to train. The value defaults to 1.
        :param batch_size: Number of batch size you want to train. The value defaults to 32.
               if you input a pytorch dataloader for `data`, the batch_size will follow the
               batch_size setted in `data`.if the forecaster is distributed, the batch_size will be
               evenly distributed to all workers.

        :return: Evaluation results on data.
        """
        # input transform
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

            # numpy data shape checking
            if isinstance(data, tuple):
                check_data(data[0], data[1], self.data_config)
            else:
                warnings.warn("Data shape checking is not supported by dataloader input.")

            # data transformation
            if isinstance(data, tuple):
                if batch_size % self.num_processes != 0:
                    warnings.warn("'batch_size' cannot be divided with no remainder by "
                                  "'self.num_processes'. We got 'batch_size' = {} and "
                                  "'self.num_processes' = {}".
                                  format(batch_size, self.num_processes))
                data = DataLoader(TensorDataset(torch.from_numpy(data[0]),
                                                torch.from_numpy(data[1])),
                                  batch_size=max(1, batch_size//self.num_processes),
                                  shuffle=True)

            # Trainer init and fitting
            self.trainer = Trainer(logger=False, max_epochs=epochs,
                                   checkpoint_callback=self.checkpoint_callback,
                                   num_processes=self.num_processes, use_ipex=self.use_ipex,
                                   distributed_backend="spawn")
            self.trainer.fit(self.internal, data)
            self.fitted = True

    def predict(self, data, batch_size=32, quantize=False):
        """
        Predict using a trained forecaster.

        if you want to predict on a single node(which is common practice), please call
        .to_local().predict(x, ...)

        :param data: The data support following formats:

               | 1. a numpy ndarray x:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | 2. a xshard item:
               | each partition can be a dictionary of {'x': x}, where x's shape
               | should follow the shape stated before.
               | 3. pytorch dataloader:
               | the dataloader needs to return at least x in each iteration
               | with the shape as following:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | If returns x and y only get x.

        :param batch_size: predict batch size. The value will not affect predict
               result but will affect resources cost(e.g. memory and time).
        :param quantize: if use the quantized model to predict.

        :return: A numpy array with shape (num_samples, horizon, target_dim)
                 if data is a numpy ndarray or a dataloader.
                 A xshard item with format {'prediction': result},
                 where result is a numpy array with shape (num_samples, horizon, target_dim)
                 if data is a xshard item.
        """
        from bigdl.chronos.pytorch.utils import _pytorch_fashion_inference

        # data transform
        is_local_data = isinstance(data, (np.ndarray, DataLoader))
        if is_local_data and self.distributed:
            if isinstance(data, DataLoader):
                from bigdl.nano.utils.log4Error import invalidInputError
                invalidInputError(False,
                                  "We will be support input dataloader later.")
            data = np_to_xshard(data)
        if not is_local_data and not self.distributed:
            data = xshard_to_np(data, mode="predict")

        if self.distributed:
            yhat = self.internal.predict(data, batch_size=batch_size)
            if is_local_data:
                expand_dim = []
                if self.data_config["future_seq_len"] == 1:
                    expand_dim.append(1)
                if self.data_config["output_feature_num"] == 1:
                    expand_dim.append(2)
                yhat = xshard_to_np(yhat, mode="yhat", expand_dim=expand_dim)
            return yhat
        else:
            if not self.fitted:
                from bigdl.nano.utils.log4Error import invalidInputError
                invalidInputError(False,
                                  "You must call fit or restore first before calling predict!")
            if quantize:
                yhat = _pytorch_fashion_inference(model=self.pytorch_int8,
                                                  input_data=data,
                                                  batch_size=batch_size)
            else:
                self.internal.eval()
                yhat = _pytorch_fashion_inference(model=self.internal,
                                                  input_data=data,
                                                  batch_size=batch_size)
            if not is_local_data:
                yhat = np_to_xshard(yhat, prefix="prediction")
            return yhat

    def predict_with_onnx(self, data, batch_size=32, quantize=False):
        """
        Predict using a trained forecaster with onnxruntime. The method can only be
        used when forecaster is a non-distributed version.

        Directly call this method without calling build_onnx is valid and Forecaster will
        automatically build an onnxruntime session with default settings.

        :param data: The data support following formats:

               | 1. a numpy ndarray x:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | 2. pytorch dataloader:
               | the dataloader needs to return at least x in each iteration
               | with the shape as following:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | If returns x and y only get x.

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
        if quantize:
            return _pytorch_fashion_inference(model=self.onnxruntime_int8,
                                              input_data=data,
                                              batch_size=batch_size)
        else:
            if self.onnxruntime_fp32 is None:
                self.build_onnx()
            return _pytorch_fashion_inference(model=self.onnxruntime_fp32,
                                              input_data=data,
                                              batch_size=batch_size)

    def predict_with_openvino(self, data, batch_size=32):
        """
        Predict using a trained forecaster with openvino. The method can only be
        used when forecaster is a non-distributed version.

        Directly call this method without calling build_openvino is valid and Forecaster will
        automatically build an openvino session with default settings.

        :param data: The data support following formats:

               | 1. a numpy ndarray x:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.

        :param batch_size: predict batch size. The value will not affect predict
               result but will affect resources cost(e.g. memory and time). Defaults
               to 32. None for all-data-single-time inference.

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
        if self.openvino_fp32 is None:
            self.build_openvino()
        return _pytorch_fashion_inference(model=self.openvino_fp32,
                                          input_data=data,
                                          batch_size=batch_size)

    def evaluate(self, data, batch_size=32, multioutput="raw_values", quantize=False):
        """
        Evaluate using a trained forecaster.

        Please note that evaluate result is calculated by scaled y and yhat. If you scaled
        your data (e.g. use .scale() on the TSDataset) please follow the following code
        snap to evaluate your result if you need to evaluate on unscaled data.

        if you want to evaluate on a single node(which is common practice), please call
        .to_local().evaluate(data, ...)

        >>> from bigdl.orca.automl.metrics import Evaluator
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
               | 2. a xshard item:
               | each partition can be a dictionary of {'x': x, 'y': y}, where x and y's shape
               | should follow the shape stated before.
               | 3. pytorch dataloader:
               | the dataloader should return x, y in each iteration with the shape as following:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.

        :param batch_size: evaluate batch size. The value will not affect evaluate
               result but will affect resources cost(e.g. memory and time).
        :param multioutput: Defines aggregating of multiple output values.
               String in ['raw_values', 'uniform_average']. The value defaults to
               'raw_values'.The param is only effective when the forecaster is a
               non-distribtued version.
        :param quantize: if use the quantized model to predict.

        :return: A list of evaluation results. Each item represents a metric.
        """
        from bigdl.chronos.pytorch.utils import _pytorch_fashion_inference

        # data transform
        is_local_data = isinstance(data, (tuple, DataLoader))
        if not is_local_data and not self.distributed:
            data = xshard_to_np(data, mode="fit")
        if self.distributed:
            data = np_to_creator(data) if is_local_data else data
            return self.internal.evaluate(data=data,
                                          batch_size=batch_size)
        else:
            if not self.fitted:
                from bigdl.nano.utils.log4Error import invalidInputError
                invalidInputError(False,
                                  "You must call fit or restore first before calling evaluate!")
            if isinstance(data, DataLoader):
                input_data = data
                target = np.concatenate(tuple(val[1] for val in data), axis=0)
            else:
                input_data, target = data
            if quantize:
                yhat = _pytorch_fashion_inference(model=self.pytorch_int8,
                                                  input_data=input_data,
                                                  batch_size=batch_size)
            else:
                self.internal.eval()
                yhat = _pytorch_fashion_inference(model=self.internal,
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
        automatically build an onnxruntime session with default settings.

        Please note that evaluate result is calculated by scaled y and yhat. If you scaled
        your data (e.g. use .scale() on the TSDataset) please follow the following code
        snap to evaluate your result if you need to evaluate on unscaled data.

        >>> from bigdl.orca.automl.metrics import Evaluator
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
               | 2. pytorch dataloader:
               | should be the same as future_seq_len and output_feature_num.
               | the dataloader should return x, y in each iteration with the shape as following:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim

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
        if isinstance(data, DataLoader):
            input_data = data
            target = np.concatenate(tuple(val[1] for val in data), axis=0)
        else:
            input_data, target = data
        if quantize:
            yhat = _pytorch_fashion_inference(model=self.onnxruntime_int8,
                                              input_data=input_data,
                                              batch_size=batch_size)
        else:
            if self.onnxruntime_fp32 is None:
                self.build_onnx()
            yhat = _pytorch_fashion_inference(model=self.onnxruntime_fp32,
                                              input_data=input_data,
                                              batch_size=batch_size)

        aggregate = 'mean' if multioutput == 'uniform_average' else None
        return Evaluator.evaluate(self.metrics, target, yhat, aggregate=aggregate)

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
                try:
                    Trainer.save(self.pytorch_int8, quantize_checkpoint_file)
                except RuntimeError:
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
            from bigdl.nano.pytorch.lightning import LightningModuleFromTorch
            from bigdl.chronos.pytorch import TSTrainer as Trainer

            model = self.model_creator({**self.model_config, **self.data_config})
            loss = self.loss_creator(self.loss_config)
            optimizer = self.optimizer_creator(model, self.optim_config)
            self.internal = LightningModuleFromTorch.load_from_checkpoint(checkpoint_file,
                                                                          model=model,
                                                                          loss=loss,
                                                                          optimizer=optimizer)
            self.internal = Trainer.compile(self.internal)
            self.fitted = True
            if quantize_checkpoint_file:
                # self.internal.load_quantized_state_dict(torch.load(quantize_checkpoint_file))
                self.pytorch_int8 = Trainer.load(quantize_checkpoint_file,
                                                 self.internal)
            # This trainer is only for quantization, once the user call `fit`, it will be
            # replaced according to the new training config
            self.trainer = Trainer(logger=False, max_epochs=1,
                                   checkpoint_callback=self.checkpoint_callback,
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
                               checkpoint_callback=self.checkpoint_callback,
                               num_processes=self.num_processes, use_ipex=self.use_ipex)

        self.distributed = False
        self.fitted = True
        self.onnxruntime_fp32 = None  # onnxruntime session for fp32 precision
        self.openvino_fp32 = None  # openvino session for fp32 precision
        self.onnxruntime_int8 = None  # onnxruntime session for int8 precision
        self.pytorch_int8 = None  # pytorch model for int8 precision
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

    def build_onnx(self, thread_num=None, sess_options=None):
        '''
        Build onnx model to speed up inference and reduce latency.
        The method is Not required to call before predict_with_onnx,
        evaluate_with_onnx or export_onnx_file.
        It is recommended to use when you want to:

        | 1. Strictly control the thread to be used during inferencing.
        | 2. Alleviate the cold start problem when you call predict_with_onnx
             for the first time.

        :param thread_num: int, the num of thread limit. The value is set to None by
               default where no limit is set.
        :param sess_options: an onnxruntime.SessionOptions instance, if you set this
               other than None, a new onnxruntime session will be built on this setting
               and ignore other settings you assigned(e.g. thread_num...).

        Example:
            >>> # to pre build onnx sess
            >>> forecaster.build_onnx(thread_num=1)  # build onnx runtime sess for single thread
            >>> pred = forecaster.predict_with_onnx(data)
            >>> # ------------------------------------------------------
            >>> # directly call onnx related method is also supported
            >>> pred = forecaster.predict_with_onnx(data)
        '''
        import onnxruntime
        from bigdl.chronos.pytorch import TSTrainer as Trainer
        from bigdl.nano.utils.log4Error import invalidInputError
        if sess_options is not None and not isinstance(sess_options, onnxruntime.SessionOptions):
            invalidInputError(False,
                              "sess_options should be an onnxruntime.SessionOptions instance"
                              f", but found {type(sess_options)}")
        if sess_options is None:
            sess_options = onnxruntime.SessionOptions()
            if thread_num is not None:
                sess_options.intra_op_num_threads = thread_num
        if self.distributed:
            invalidInputError(False,
                              "build_onnx has not been supported for distributed "
                              "forecaster. You can call .to_local() to transform the "
                              "forecaster to a non-distributed version.")
        dummy_input = torch.rand(1, self.data_config["past_seq_len"],
                                 self.data_config["input_feature_num"])
        self.onnxruntime_fp32 = Trainer.trace(self.internal,
                                              input_sample=dummy_input,
                                              accelerator="onnxruntime",
                                              onnxruntime_session_options=sess_options)

    def build_openvino(self):
        '''
        Build openvino model to speed up inference and reduce latency.
        The method is Not required to call before predict_with_openvino.
        '''
        from bigdl.chronos.pytorch import TSTrainer as Trainer
        from bigdl.nano.utils.log4Error import invalidInputError

        if self.distributed:
            invalidInputError(False,
                              "build_openvino has not been supported for distributed "
                              "forecaster. You can call .to_local() to transform the "
                              "forecaster to a non-distributed version.")
        dummy_input = torch.rand(1, self.data_config["past_seq_len"],
                                 self.data_config["input_feature_num"])
        self.openvino_fp32 = Trainer.trace(self.internal,
                                           input_sample=dummy_input,
                                           accelerator="openvino")

    def export_onnx_file(self, dirname="model.onnx", quantized_dirname="qmodel.onnx"):
        """
        Save the onnx model file to the disk.

        :param dirname: The dir location you want to save the onnx file.
        """
        from bigdl.chronos.pytorch import TSTrainer as Trainer
        from bigdl.nano.utils.log4Error import invalidInputError
        if self.distributed:
            invalidInputError(False,
                              "export_onnx_file has not been supported for distributed "
                              "forecaster. You can call .to_local() to transform the "
                              "forecaster to a non-distributed version.")
        dummy_input = torch.rand(1, self.data_config["past_seq_len"],
                                 self.data_config["input_feature_num"])
        if quantized_dirname:
            Trainer.save(self.onnxruntime_int8, dirname)
        if dirname:
            if self.onnxruntime_fp32 is None:
                self.build_onnx()
            Trainer.save(self.onnxruntime_fp32, dirname)

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
                 max_trials=1):
        """
        Quantize the forecaster.

        :param calib_data: A torch.utils.data.dataloader.DataLoader object for calibration.
               Required for static quantization.
        :param val_data: A torch.utils.data.dataloader.DataLoader object for evaluation.
        :param metric: A str represent the metrics for tunning the quality of
               quantization. You may choose from "mse", "mae", "rmse", "r2", "mape", "smape".
        :param conf: A path to conf yaml file for quantization. Default to None,
               using default config.
        :param framework: string or list, [{'pytorch'|'pytorch_fx'|'pytorch_ipex'},
               {'onnxrt_integerops'|'onnxrt_qlinearops'}]. Default: 'pytorch_fx'.
               Consistent with Intel Neural Compressor.
        :param approach: str, 'static' or 'dynamic'. Default to 'static'.
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
        """
        # check model support for quantization
        from bigdl.nano.utils.log4Error import invalidInputError
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
        framework = [framework] if isinstance(framework, str) else framework
        temp_quantized_model = None
        for framework_item in framework:
            accelerator, method = framework_item.split('_')
            if accelerator == 'pytorch':
                accelerator = None
            else:
                accelerator = 'onnxruntime'
                method = method[:-3]
            q_model = self.trainer.quantize(self.internal,
                                            precision='int8',
                                            accelerator=accelerator,
                                            method=method,
                                            calib_dataloader=calib_data,
                                            metric=metric,
                                            conf=conf,
                                            approach=approach,
                                            tuning_strategy=tuning_strategy,
                                            accuracy_criterion=accuracy_criterion,
                                            timeout=timeout,
                                            max_trials=max_trials)
            if accelerator == 'onnxruntime':
                self.onnxruntime_int8 = q_model
            if accelerator is None:
                self.pytorch_int8 = q_model


def _str2metric(metric):
    # map metric str to function
    if isinstance(metric, str):
        from bigdl.chronos.metric.forecast_metrics import TORCHMETRICS_REGRESSION_MAP
        metric = TORCHMETRICS_REGRESSION_MAP[metric]
    return metric
