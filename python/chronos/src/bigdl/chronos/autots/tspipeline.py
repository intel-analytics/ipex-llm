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

import os
from bigdl.chronos.utils import LazyImport
torch = LazyImport('torch')
import types
import numpy as np

from bigdl.chronos.data import TSDataset
from bigdl.chronos.metric.forecast_metrics import Evaluator

DEFAULT_MODEL_INIT_DIR = "model_init.ckpt"
DEFAULT_BEST_MODEL_DIR = "best_model.ckpt"
DEFAULT_DATA_PROCESS_DIR = "data_process.ckpt"
DEFAULT_BEST_CONFIG_DIR = "best_config.ckpt"


class TSPipeline:
    '''
    TSPipeline is an E2E solution for time series analysis (only forecasting task for now).
    You can use TSPipeline to:

    1. Further development on the prototype. (predict, evaluate, incremental fit)

    2. Deploy the model to their scenario. (save, load)
    '''
    def __init__(self,
                 model,
                 loss,
                 optimizer,
                 model_creator,
                 loss_creator,
                 optimizer_creator,
                 best_config,
                 **kwargs):
        from bigdl.nano.pytorch.trainer import Trainer

        # for runtime fit/predict/evaluate
        self._best_model = Trainer.compile(model=model,
                                           loss=loss,
                                           optimizer=optimizer)
        self._best_config = best_config
        self._onnxruntime_fp32 = None
        self._onnxruntime_int8 = None
        self._pytorch_int8 = None

        # for data postprocessing
        self._scaler = None
        self._scaler_index = None
        if "scaler" in kwargs.keys():
            self._scaler = kwargs["scaler"]
            self._scaler_index = kwargs["scaler_index"]

        # for save/load
        self.model_creator = model_creator
        self.loss_creator = loss_creator
        self.optimizer_creator = optimizer_creator

    def evaluate(self, data, metrics=['mse'], multioutput="uniform_average",
                 batch_size=32, quantize=False):
        '''
        Evaluate the time series pipeline.

        :param data: data can be a TSDataset or data creator.
               The TSDataset should follow the same operations as the training
               TSDataset used in AutoTSEstimator.fit.
        :param metrics: list of string or callable. e.g. ['mse'] or [customized_metrics]
               If callable function, it signature should be func(y_true, y_pred), where y_true and
               y_pred are numpy ndarray. The function should return a float value as evaluation
               result.
        :param multioutput: Defines aggregating of multiple output values.
               String in ['raw_values', 'uniform_average']. The value defaults to
               'uniform_average'.
        :param batch_size: predict batch_size, the process will cost more time
               if batch_size is small while cost less memory. The param is only
               effective when data is a TSDataset. The values defaults to 32.
        :param quantize: if use the quantized model to predict.
        '''
        from bigdl.chronos.pytorch.utils import _pytorch_fashion_inference

        # predict
        if isinstance(data, TSDataset):
            x, y = self._tsdataset_to_numpy(data, is_predict=False)
            if quantize:
                yhat = _pytorch_fashion_inference(model=self._pytorch_int8,
                                                  input_data=x,
                                                  batch_size=batch_size)
            else:
                self._best_model.eval()
                yhat = _pytorch_fashion_inference(model=self._best_model,
                                                  input_data=x,
                                                  batch_size=batch_size)
            # unscale
            yhat = self._tsdataset_unscale(yhat)
            y = self._tsdataset_unscale(y)
        elif isinstance(data, types.FunctionType):
            yhat_list, y_list = [], []
            self._best_config.update({'batch_size': batch_size})
            for x, y in data(self._best_config):
                if quantize:
                    yhat = _pytorch_fashion_inference(model=self._pytorch_int8,
                                                      input_data=x.numpy())
                else:
                    self._best_model.eval()
                    yhat = _pytorch_fashion_inference(model=self._best_model,
                                                      input_data=x.numpy())
                yhat_list.append(yhat)
                y_list.append(y)
            yhat = np.concatenate(yhat_list, axis=0)
            y = torch.cat(y_list, dim=0).numpy()
        else:
            from bigdl.nano.utils.log4Error import invalidInputError
            invalidInputError(False,
                              "We only support input tsdataset or data creator, "
                              f"but found {data.__class__.__name__}.")

        # evaluate
        aggregate = 'mean' if multioutput == 'uniform_average' else None
        eval_result = Evaluator.evaluate(metrics, y, yhat, aggregate=aggregate)
        return eval_result

    def evaluate_with_onnx(self, data, metrics=['mse'], multioutput="uniform_average",
                           batch_size=32, quantize=False):
        '''
        Evaluate the time series pipeline with onnx.

        :param data: data can be a TSDataset or data creator.
               The TSDataset should follow the same operations as the training
               TSDataset used in AutoTSEstimator.fit.
        :param metrics: list of string or callable. e.g. ['mse'] or [customized_metrics]
               If callable function, it signature should be func(y_true, y_pred), where y_true and
               y_pred are numpy ndarray. The function should return a float value as evaluation
               result.
        :param multioutput: Defines aggregating of multiple output values.
               String in ['raw_values', 'uniform_average']. The value defaults to
               'uniform_average'.
        :param batch_size: predict batch_size, the process will cost more time
               if batch_size is small while cost less memory. The param is only
               effective when data is a TSDataset. The values defaults to 32.
        :param quantize: if use the quantized model to predict.
        '''
        from bigdl.chronos.pytorch import TSTrainer as Trainer
        from bigdl.chronos.pytorch.utils import _pytorch_fashion_inference
        from bigdl.nano.utils.log4Error import invalidInputError
        # predict with onnx
        if isinstance(data, TSDataset):
            x, y = self._tsdataset_to_numpy(data, is_predict=False)
            yhat = None
            if quantize:
                yhat = _pytorch_fashion_inference(model=self._onnxruntime_int8,
                                                  input_data=x,
                                                  batch_size=batch_size)
            else:
                if self._onnxruntime_fp32 is None:
                    self._onnxruntime_fp32 = Trainer.trace(self._best_model,
                                                           input_sample=torch.from_numpy(x[0:1]),
                                                           accelerator="onnxruntime")
                yhat = _pytorch_fashion_inference(model=self._onnxruntime_fp32,
                                                  input_data=x,
                                                  batch_size=batch_size)
            yhat = self._tsdataset_unscale(yhat)
            # unscale
            y = self._tsdataset_unscale(y)
        elif isinstance(data, types.FunctionType):
            yhat_list, y_list = [], []
            self._best_config.update({'batch_size': batch_size})
            yhat = None
            for x, y in data(self._best_config):
                if quantize:
                    yhat = _pytorch_fashion_inference(model=self._onnxruntime_int8,
                                                      input_data=x.numpy(),
                                                      batch_size=batch_size)
                else:
                    if self._onnxruntime_fp32 is None:
                        self._onnxruntime_fp32 = Trainer.trace(self._best_model,
                                                               input_sample=x[0:1],
                                                               accelerator="onnxruntime")
                    yhat = _pytorch_fashion_inference(model=self._onnxruntime_fp32,
                                                      input_data=x.numpy(),
                                                      batch_size=batch_size)
                yhat_list.append(yhat)
                y_list.append(y)
            yhat = np.concatenate(yhat_list, axis=0)
            y = torch.cat(y_list, dim=0).numpy()
        else:
            invalidInputError(False,
                              "We only support input tsdataset or data creator, "
                              f"but found {data.__class__.__name__}.")
        # evaluate
        aggregate = 'mean' if multioutput == 'uniform_average' else None
        eval_result = Evaluator.evaluate(metrics, y, yhat, aggregate=aggregate)
        return eval_result

    def predict(self, data, batch_size=32, quantize=False):
        '''
        Rolling predict with time series pipeline.

        :param data: data can be a TSDataset or data creator.
               The TSDataset should follow the same operations as the training
               TSDataset used in AutoTSEstimator.fit.
        :param batch_size: predict batch_size, the process will cost more time
               if batch_size is small while cost less memory.  The param is only
               effective when data is a TSDataset. The values defaults to 32.
        :param quantize: if use the quantized model to predict.
        '''
        from bigdl.chronos.pytorch.utils import _pytorch_fashion_inference
        from bigdl.nano.utils.log4Error import invalidInputError
        if isinstance(data, TSDataset):
            x = self._tsdataset_to_numpy(data, is_predict=True)
            if quantize:
                yhat = _pytorch_fashion_inference(model=self._pytorch_int8,
                                                  input_data=x,
                                                  batch_size=batch_size)
            else:
                self._best_model.eval()
                yhat = _pytorch_fashion_inference(model=self._best_model,
                                                  input_data=x,
                                                  batch_size=batch_size)
            yhat = self._tsdataset_unscale(yhat)
        elif isinstance(data, types.FunctionType):
            yhat_list = []
            self._best_config.update({'batch_size': batch_size})
            for x, _ in data(self._best_config):
                if quantize:
                    yhat = _pytorch_fashion_inference(model=self._pytorch_int8,
                                                      input_data=x.numpy())
                else:
                    self._best_model.eval()
                    yhat = _pytorch_fashion_inference(model=self._best_model,
                                                      input_data=x.numpy())
                yhat_list.append(yhat)
            yhat = np.concatenate(yhat_list, axis=0)
        else:
            invalidInputError(False,
                              "We only support input tsdataset or data creator, "
                              f"but found {data.__class__.__name__}")
        return yhat

    def predict_with_onnx(self, data, batch_size=32, quantize=False):
        '''
        Rolling predict with onnx with time series pipeline.

        :param data: data can be a TSDataset or data creator.
               The TSDataset should follow the same operations as the training
               TSDataset used in AutoTSEstimator.fit.
        :param batch_size: predict batch_size, the process will cost more time
               if batch_size is small while cost less memory.  The param is only
               effective when data is a TSDataset. The values defaults to 32.
        :param quantize: if use the quantized model to predict.
        '''
        from bigdl.chronos.pytorch import TSTrainer as Trainer
        from bigdl.chronos.pytorch.utils import _pytorch_fashion_inference
        from bigdl.nano.utils.log4Error import invalidInputError
        if isinstance(data, TSDataset):
            x = self._tsdataset_to_numpy(data, is_predict=True)
            yhat = None
            if quantize:
                yhat = _pytorch_fashion_inference(model=self._onnxruntime_int8,
                                                  input_data=x,
                                                  batch_size=batch_size)
            else:
                if self._onnxruntime_fp32 is None:
                    self._onnxruntime_fp32 = Trainer.trace(self._best_model,
                                                           input_sample=torch.from_numpy(x[0:1]),
                                                           accelerator="onnxruntime")
                yhat = _pytorch_fashion_inference(model=self._onnxruntime_fp32,
                                                  input_data=x,
                                                  batch_size=batch_size)
            yhat = self._tsdataset_unscale(yhat)
        elif isinstance(data, types.FunctionType):
            yhat = None
            yhat_list = []
            self._best_config.update({'batch_size': batch_size})
            for x, _ in data(self._best_config):
                if quantize:
                    yhat = _pytorch_fashion_inference(model=self._onnxruntime_int8,
                                                      input_data=x.numpy(),
                                                      batch_size=batch_size)
                else:
                    if self._onnxruntime_fp32 is None:
                        self._onnxruntime_fp32 = Trainer.trace(self._best_model,
                                                               input_sample=x[0:1],
                                                               accelerator="onnxruntime")
                    yhat = _pytorch_fashion_inference(model=self._onnxruntime_fp32,
                                                      input_data=x.numpy(),
                                                      batch_size=batch_size)
                yhat_list.append(yhat)
            yhat = np.concatenate(yhat_list, axis=0)
        else:
            invalidInputError(False,
                              "We only support input tsdataset or data creator, "
                              f"but found {data.__class__.__name__}")
        return yhat

    def fit(self,
            data,
            validation_data=None,
            epochs=1,
            batch_size=None,
            **kwargs):
        '''
        Incremental fitting

        :param data: The data support following formats:

               | 1. data creator:
               | a function that takes a config dictionary as parameter and
               | returns a PyTorch DataLoader.
               |
               | 2. a bigdl.chronos.data.TSDataset:
               | the TSDataset should follow the same operations as the training
               | TSDataset used in `AutoTSEstimator.fit`.

        :param validation_data: validation data, same format as data.
        :param epochs: incremental fitting epoch. The value defaults to 1.
        :param metric: evaluate metric.
        :param batch_size: batch size, defaults to None, which takes the searched best batch_size.
        :param **kwargs: args to be passed to bigdl-nano trainer.
        '''
        from bigdl.chronos.pytorch import TSTrainer as Trainer
        from bigdl.nano.utils.log4Error import invalidInputError
        train_loader = None
        valid_loader = None
        if isinstance(data, TSDataset):
            if batch_size is None:
                batch_size = self._best_config["batch_size"]
            train_loader = self._tsdataset_to_loader(data, batch_size=batch_size)
            if validation_data:
                valid_loader = self._tsdataset_to_loader(validation_data, batch_size=batch_size)
        elif isinstance(data, types.FunctionType):
            if batch_size:
                self._best_config.update({'batch_size': batch_size})
            train_loader = data(self._best_config)
            if validation_data:
                valid_loader = validation_data(self._best_config)
        else:
            invalidInputError(False,
                              "We only support input TSDataset or data creator, "
                              f"but found {data.__class__.__name__}.")

        self.trainer = Trainer(max_epochs=epochs, **kwargs)
        self.trainer.fit(self._best_model,
                         train_dataloaders=train_loader,
                         val_dataloaders=valid_loader)

    def save(self, file_path):
        '''
        Save the TSPipeline to a folder

        :param file_path: the folder location to save the pipeline
        '''
        import pickle
        if not os.path.isdir(file_path):
            os.mkdir(file_path)
        model_init_path = os.path.join(file_path, DEFAULT_MODEL_INIT_DIR)
        model_path = os.path.join(file_path, DEFAULT_BEST_MODEL_DIR)
        data_process_path = os.path.join(file_path, DEFAULT_DATA_PROCESS_DIR)
        best_config_path = os.path.join(file_path, DEFAULT_BEST_CONFIG_DIR)
        model_init = {"model_creator": self.model_creator,
                      "optimizer_creator": self.optimizer_creator,
                      "loss_creator": self.loss_creator}
        data_process = {"scaler": self._scaler,
                        "scaler_index": self._scaler_index}
        with open(model_init_path, "wb") as f:
            pickle.dump(model_init, f)
        with open(data_process_path, "wb") as f:
            pickle.dump(data_process, f)
        with open(best_config_path, "wb") as f:
            pickle.dump(self._best_config, f)
        # self._best_model.save(model_path)
        torch.save(self._best_model.model.state_dict(), model_path)

    @staticmethod
    def load(file_path):
        '''
        Load the TSPipeline to a folder

        :param file_path: the folder location to load the pipeline
        '''
        import pickle
        model_init_path = os.path.join(file_path, DEFAULT_MODEL_INIT_DIR)
        model_path = os.path.join(file_path, DEFAULT_BEST_MODEL_DIR)
        data_process_path = os.path.join(file_path, DEFAULT_DATA_PROCESS_DIR)
        best_config_path = os.path.join(file_path, DEFAULT_BEST_CONFIG_DIR)
        with open(model_init_path, "rb") as f:
            model_init = pickle.load(f)
        with open(data_process_path, "rb") as f:
            data_process = pickle.load(f)
        with open(best_config_path, "rb") as f:
            best_config = pickle.load(f)

        model_creator = model_init["model_creator"]
        optimizer_creator = model_init["optimizer_creator"]
        loss_creator = model_init["loss_creator"]

        model = model_creator(best_config)
        model.load_state_dict(torch.load(model_path))

        if isinstance(optimizer_creator, types.FunctionType):
            optimizer = optimizer_creator(model, best_config)
        else:
            optimizer = optimizer_creator(model.parameters(),
                                          lr=best_config.get('lr', 0.001))

        if isinstance(loss_creator, torch.nn.modules.loss._Loss):
            loss = loss_creator
        else:
            loss = loss_creator(best_config)

        return TSPipeline(model=model,
                          loss=loss,
                          optimizer=optimizer,
                          model_creator=model_creator,
                          loss_creator=loss_creator,
                          optimizer_creator=optimizer_creator,
                          best_config=best_config,
                          **data_process)

    def quantize(self,
                 calib_data,
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
        Quantization TSPipeline.

        :param calib_data: Required for static quantization or evaluation.

               | 1. data creator:
               | a function that takes a config dictionary as parameter and
               | returns a PyTorch DataLoader.
               |
               | 2. a bigdl.chronos.data.TSDataset:
               | the TSDataset should follow the same operations as the training
               | TSDataset used in `AutoTSEstimator.fit`.
               |
               | 3. A torch.utils.data.dataloader.DataLoader object for calibration,
               | Users should set the configs correctly (e.g. past_seq_len, ...).
               | They can be found in TSPipeline._best_config.
               |
               | 4. A numpy ndarray tuple (x, y).
               | x's shape is (num_samples, past_seq_len, input_feature_dim).
               | y's shape is (num_samples, future_seq_len, output_feature_dim).
               | They can be found in TSPipeline._best_config.

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
        from torch.utils.data import DataLoader, TensorDataset
        from bigdl.chronos.data import TSDataset
        from bigdl.nano.utils.log4Error import invalidInputError
        # check model support for quantization
        from bigdl.chronos.autots.utils import check_quantize_available
        check_quantize_available(self._best_model.model)
        # calib data should be set if the forecaster is just loaded
        if calib_data is None and approach.startswith("static"):
            invalidInputError(False,
                              "You must set a `calib_data` "
                              "for quantization When you use 'static'.")
        elif calib_data and approach.startswith("dynamic"):
            invalidInputError(False,
                              "`calib_data` should be None When you use 'dynamic'.")

        # preprocess data.
        from .utils import preprocess_quantize_data
        calib_data = preprocess_quantize_data(self, calib_data)

        # map metric str to function
        from bigdl.chronos.metric.forecast_metrics import REGRESSION_MAP
        if isinstance(metric, str):
            metric_func = REGRESSION_MAP[metric]

            def metric(y_label, y_predict):
                y_label = y_label.numpy()
                y_predict = y_predict.numpy()
                return metric_func(y_label, y_predict)

        # init acc criterion
        accuracy_criterion = None
        if relative_drop and absolute_drop:
            invalidInputError(False, "Please unset either `relative_drop` or `absolute_drop`.")
        if relative_drop:
            accuracy_criterion = {'relative': relative_drop, 'higher_is_better': False}
        if absolute_drop:
            accuracy_criterion = {'absolute': absolute_drop, 'higher_is_better': False}

        from bigdl.nano.pytorch.trainer import Trainer
        self._trainer = Trainer(logger=False, max_epochs=1,
                                checkpoint_callback=False,
                                use_ipex=False)

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
            q_model = self._trainer.quantize(self._best_model,
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
            if accelerator == "onnxruntime":
                self._onnxruntime_int8 = q_model
            if accelerator is None:
                self._pytorch_int8 = q_model

    def _tsdataset_to_loader(self, data, is_predict=False, batch_size=32):
        self._check_mixed_data_type_usage()
        lookback = self._best_config["past_seq_len"]
        horizon = 0 if is_predict else self._best_config["future_seq_len"]
        selected_features = self._best_config["selected_features"]
        data_loader = data.to_torch_data_loader(batch_size=batch_size,
                                                lookback=lookback,
                                                horizon=horizon,
                                                feature_col=selected_features)
        return data_loader

    def _tsdataset_to_numpy(self, data, is_predict=False):
        self._check_mixed_data_type_usage()
        lookback = self._best_config["past_seq_len"]
        horizon = self._best_config["future_seq_len"]
        selected_features = self._best_config["selected_features"]
        data.roll(lookback=lookback,
                  horizon=horizon,
                  feature_col=selected_features,
                  is_predict=is_predict)
        return data.to_numpy()

    def _check_mixed_data_type_usage(self):
        from bigdl.nano.utils.log4Error import invalidInputError
        for key in ("past_seq_len", "future_seq_len", "selected_features"):
            if key not in self._best_config:
                invalidInputError(False,
                                  "You use a data creator to fit your AutoTSEstimator, "
                                  "and use a TSDataset to predict/evaluate/fit on the TSPipeline."
                                  "Please stick to the same data type.")

    def _tsdataset_unscale(self, y):
        if self._scaler:
            from bigdl.chronos.data.utils.scale import unscale_timeseries_numpy
            y = unscale_timeseries_numpy(y, self._scaler, self._scaler_index)
        return y
