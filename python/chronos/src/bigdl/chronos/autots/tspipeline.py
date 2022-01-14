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
import torch
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
                                           optimizer=optimizer,
                                           onnx=True)
        self._best_config = best_config

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

    def evaluate(self, data, metrics=['mse'], multioutput="uniform_average", batch_size=32):
        '''
        Evaluate the time series pipeline.

        :param data: data can be a TSDataset or data creator.
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
        if isinstance(data, TSDataset):
            x, y = self._tsdataset_to_numpy(data, is_predict=False)
            yhat = self._best_model.inference(torch.from_numpy(x),
                                              batch_size=batch_size,
                                              backend=None).numpy()
            # unscale
            yhat = self._tsdataset_unscale(yhat)
            y = self._tsdataset_unscale(y)
        elif isinstance(data, types.FunctionType):
            yhat_list, y_list = [], []
            self._best_config.update({'batch_size': batch_size})
            for x, y in data(self._best_config):
                yhat = self._best_model.inference(x, backend=None)
                yhat_list.append(yhat)
                y_list.append(y)
            yhat = torch.cat(yhat_list, dim=0).numpy()
            y = torch.cat(y_list, dim=0).numpy()
        else:
            raise RuntimeError("We only support input tsdataset or data creator, "
                               f"but found {data.__class__.__name__}.")

        # evaluate
        aggregate = 'mean' if multioutput == 'uniform_average' else None
        eval_result = Evaluator.evaluate(metrics, y, yhat, aggregate=aggregate)
        return eval_result

    def evaluate_with_onnx(self, data, metrics=['mse'], multioutput="uniform_average",
                           batch_size=32):
        '''
        Evaluate the time series pipeline with onnx.

        :param data: data can be a TSDataset or data creator.
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
        if isinstance(data, TSDataset):
            x, y = self._tsdataset_to_numpy(data, is_predict=False)
            yhat = self._best_model.inference(x,
                                              batch_size=batch_size,
                                              backend="onnx")
            yhat = self._tsdataset_unscale(yhat)
            # unscale
            y = self._tsdataset_unscale(y)
        elif isinstance(data, types.FunctionType):
            yhat_list, y_list = [], []
            self._best_config.update({'batch_size': batch_size})
            for x, y in data(self._best_config):
                yhat = self._best_model.inference(x.numpy(), backend="onnx")
                yhat_list.append(yhat)
                y_list.append(y)
            yhat = np.concatenate(yhat_list, axis=0)
            y = torch.cat(y_list, dim=0).numpy()
        else:
            raise RuntimeError("We only support input tsdataset or data creator, "
                               f"but found {data.__class__.__name__}.")
        # evaluate
        aggregate = 'mean' if multioutput == 'uniform_average' else None
        eval_result = Evaluator.evaluate(metrics, y, yhat, aggregate=aggregate)
        return eval_result

    def predict(self, data, batch_size=32):
        '''
        Rolling predict with time series pipeline.

        :param data: data can be a TSDataset or data creator.
               The TSDataset should follow the same operations as the training
               TSDataset used in AutoTSEstimator.fit.
        :param batch_size: predict batch_size, the process will cost more time
               if batch_size is small while cost less memory.  The param is only
               effective when data is a TSDataset. The values defaults to 32.
        '''
        if isinstance(data, TSDataset):
            x, _ = self._tsdataset_to_numpy(data, is_predict=True)
            yhat = self._best_model.inference(torch.from_numpy(x),
                                              batch_size=batch_size,
                                              backend=None)
            yhat = self._tsdataset_unscale(yhat)
        elif isinstance(data, types.FunctionType):
            yhat_list = []
            self._best_config.update({'batch_size': batch_size})
            for x, _ in data(self._best_config):
                yhat = self._best_model.inference(x, backend=None)
                yhat_list.append(yhat)
            yhat = np.concatenate(yhat_list, axis=0)
        else:
            raise RuntimeError("We only support input tsdataset or data creator, "
                               f"but found {data.__class__.__name__}")
        return yhat

    def predict_with_onnx(self, data, batch_size=32):
        '''
        Rolling predict with onnx with time series pipeline.

        :param data: data can be a TSDataset or data creator.
               The TSDataset should follow the same operations as the training
               TSDataset used in AutoTSEstimator.fit.
        :param batch_size: predict batch_size, the process will cost more time
               if batch_size is small while cost less memory.  The param is only
               effective when data is a TSDataset. The values defaults to 32.
        '''
        if isinstance(data, TSDataset):
            x, _ = self._tsdataset_to_numpy(data, is_predict=True)
            yhat = self._best_model.inference(x,
                                              batch_size=batch_size,
                                              backend="onnx")
            yhat = self._tsdataset_unscale(yhat)
        elif isinstance(data, types.FunctionType):
            yhat_list = []
            self._best_config.update({'batch_size': batch_size})
            for x, _ in data(self._best_config):
                yhat = self._best_model.inference(x.numpy(), backend="onnx")
                yhat_list.append(yhat)
            yhat = np.concatenate(yhat_list, axis=0)
        else:
            raise RuntimeError("We only support input tsdataset or data creator, "
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
        from bigdl.nano.pytorch.trainer import Trainer

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
            raise RuntimeError("We only support input TSDataset or data creator, "
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

    def _tsdataset_to_loader(self, data, is_predict=False, batch_size=32):
        lookback = self._best_config["past_seq_len"]
        horizon = 0 if is_predict else self._best_config["future_seq_len"]
        selected_features = self._best_config["selected_features"]
        data_loader = data.to_torch_data_loader(batch_size=batch_size,
                                                roll=True,
                                                lookback=lookback,
                                                horizon=horizon,
                                                feature_col=selected_features)
        return data_loader

    def _tsdataset_to_numpy(self, data, is_predict=False):
        lookback = self._best_config["past_seq_len"]
        horizon = 0 if is_predict else self._best_config["future_seq_len"]
        selected_features = self._best_config["selected_features"]
        data.roll(lookback=lookback,
                  horizon=horizon,
                  feature_col=selected_features)
        return data.to_numpy()

    def _tsdataset_unscale(self, y):
        if self._scaler:
            from bigdl.chronos.data.utils.scale import unscale_timeseries_numpy
            y = unscale_timeseries_numpy(y, self._scaler, self._scaler_index)
        return y
