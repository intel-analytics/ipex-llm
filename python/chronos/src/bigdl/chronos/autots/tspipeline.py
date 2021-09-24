#
# Copyright 2018 Analytics Zoo Authors.
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

from zoo.chronos.data import TSDataset
from zoo.orca.automl.metrics import Evaluator

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
    def __init__(self, best_model, best_config, **kwargs):
        self._best_model = best_model
        self._best_config = best_config
        self._scaler = None
        self._scaler_index = None
        if "scaler" in kwargs.keys():
            self._scaler = kwargs["scaler"]
            self._scaler_index = kwargs["scaler_index"]

    def evaluate(self, data, metrics=['mse'], multioutput="uniform_average", batch_size=32):
        '''
        Evaluate the time series pipeline.

        :param data: data can be a TSDataset or data creator(will be supported).
               The TSDataset should follow the same operations as the training
               TSDataset used in AutoTSEstimator.fit.
        :param metrics: list. The evaluation metric name to optimize. e.g. ["mse"]
        :param multioutput: Defines aggregating of multiple output values.
               String in ['raw_values', 'uniform_average']. The value defaults to
               'uniform_average'.
        :param batch_size: predict batch_size, the process will cost more time
               if batch_size is small while cost less memory. The param is only
               effective when data is a TSDataset. The values defaults to 32.
        '''
        # predict
        x, y = self._tsdataset_to_numpy(data, is_predict=False)
        yhat = self._best_model.predict(x, batch_size=batch_size)
        yhat = self._tsdataset_unscale(yhat)
        # unscale
        y = self._tsdataset_unscale(y)
        # evaluate
        eval_result = [Evaluator.evaluate(m, y_true=y, y_pred=yhat,
                                          multioutput=multioutput)
                       for m in metrics]
        return eval_result

    def evaluate_with_onnx(self, data, metrics=['mse'], multioutput="uniform_average",
                           batch_size=32):
        '''
        Evaluate the time series pipeline with onnx.

        :param data: data can be a TSDataset or data creator(will be supported).
               The TSDataset should follow the same operations as the training
               TSDataset used in AutoTSEstimator.fit.
        :param metrics: list. The evaluation metric name to optimize. e.g. ["mse"]
        :param multioutput: Defines aggregating of multiple output values.
               String in ['raw_values', 'uniform_average']. The value defaults to
               'uniform_average'.
        :param batch_size: predict batch_size, the process will cost more time
               if batch_size is small while cost less memory. The param is only
               effective when data is a TSDataset. The values defaults to 32.
        '''
        # predict with onnx
        x, y = self._tsdataset_to_numpy(data, is_predict=False)
        yhat = self._best_model.predict_with_onnx(x, batch_size=batch_size)
        yhat = self._tsdataset_unscale(yhat)
        # unscale
        y = self._tsdataset_unscale(y)
        # evaluate
        eval_result = [Evaluator.evaluate(m, y_true=y, y_pred=yhat,
                                          multioutput=multioutput)
                       for m in metrics]
        return eval_result

    def predict(self, data, batch_size=32):
        '''
        Rolling predict with time series pipeline.

        :param data: data can be a TSDataset or data creator(will be supported).
               The TSDataset should follow the same operations as the training
               TSDataset used in AutoTSEstimator.fit.
        :param batch_size: predict batch_size, the process will cost more time
               if batch_size is small while cost less memory.  The param is only
               effective when data is a TSDataset. The values defaults to 32.
        '''
        x, _ = self._tsdataset_to_numpy(data, is_predict=True)
        yhat = self._best_model.predict(x, batch_size=batch_size)
        yhat = self._tsdataset_unscale(yhat)
        return yhat

    def predict_with_onnx(self, data, batch_size=32):
        '''
        Rolling predict with onnx with time series pipeline.

        :param data: data can be a TSDataset or data creator(will be supported).
               The TSDataset should follow the same operations as the training
               TSDataset used in AutoTSEstimator.fit.
        :param batch_size: predict batch_size, the process will cost more time
               if batch_size is small while cost less memory.  The param is only
               effective when data is a TSDataset. The values defaults to 32.
        '''
        x, _ = self._tsdataset_to_numpy(data, is_predict=True)
        yhat = self._best_model.predict_with_onnx(x, batch_size=batch_size)
        yhat = self._tsdataset_unscale(yhat)
        return yhat

    def fit(self, data, validation_data=None, epochs=1, metric="mse"):
        '''
        Incremental fitting

        :param data: data can be a TSDataset or data creator(will be supported).
               the TSDataset should follow the same operations as the training
               TSDataset used in AutoTSEstimator.fit.
        :param validation_data: validation data, same format as data.
        :param epochs: incremental fitting epoch. The value defaults to 1.
        :param metric: evaluate metric.
        '''
        x, y = self._tsdataset_to_numpy(data, is_predict=False)
        if validation_data is None:
            x_val, y_val = x, y
        else:
            x_val, y_val = self._tsdataset_to_numpy(validation_data, is_predict=False)

        res = self._best_model.fit_eval(data=(x, y), validation_data=(x_val, y_val), metric=metric)
        return res

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
        model_init = {"model_creator": self._best_model.model_creator,
                      "optimizer_creator": self._best_model.optimizer_creator,
                      "loss_creator": self._best_model.loss_creator}
        data_process = {"scaler": self._scaler,
                        "scaler_index": self._scaler_index}
        with open(model_init_path, "wb") as f:
            pickle.dump(model_init, f)
        with open(data_process_path, "wb") as f:
            pickle.dump(data_process, f)
        with open(best_config_path, "wb") as f:
            pickle.dump(self._best_config, f)
        self._best_model.save(model_path)

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
        from zoo.orca.automl.model.base_pytorch_model import PytorchBaseModel
        best_model = PytorchBaseModel(**model_init)
        best_model.restore(model_path)
        return TSPipeline(best_model, best_config, **data_process)

    def _tsdataset_to_numpy(self, data, is_predict=False):
        if isinstance(data, TSDataset):
            lookback = self._best_config["past_seq_len"]
            horizon = 0 if is_predict else self._best_config["future_seq_len"]
            selected_features = self._best_config["selected_features"]
            data.roll(lookback, horizon, feature_col=selected_features)
            x, y = data.to_numpy()
        else:
            raise NotImplementedError("Data creator has not been supported now.")
        return x, y

    def _tsdataset_unscale(self, y):
        if self._scaler:
            from zoo.chronos.data.utils.scale import unscale_timeseries_numpy
            y = unscale_timeseries_numpy(y, self._scaler, self._scaler_index)
        return y
