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

import torch
from bigdl.chronos.forecaster.base_forecaster import BasePytorchForecaster
from bigdl.chronos.model.tcn import model_creator, optimizer_creator, loss_creator


class TCNForecaster(BasePytorchForecaster):
    """
        Example:
            >>> #The dataset is split into x_train, x_val, x_test, y_train, y_val, y_test
            >>> # 1. Initialize Forecaster directly
            >>> forecaster = TCNForecaster(past_seq_len=24,
                                           future_seq_len=5,
                                           input_feature_num=1,
                                           output_feature_num=1,
                                           ...)
            >>> # 2. The from_dataset method can also initialize a TCNForecaster.
            >>> tsdata = TSDataset.from_pandas(df, ...)
            >>> tsdata.roll(lookback=24, horizon=5, ...)
            >>> forecaster.from_tsdataset(tsdata, past_seq_len=None, **kwargs)
            >>> forecaster.fit(tsdata, ...)
            >>> forecaster.to_local()  # if you set distributed=True
            >>> test_pred = forecaster.predict(x_test)
            >>> test_eval = forecaster.evaluate((x_test, y_test))
            >>> forecaster.save({ckpt_name})
            >>> forecaster.load({ckpt_name})
    """
    def __init__(self,
                 past_seq_len,
                 future_seq_len,
                 input_feature_num,
                 output_feature_num,
                 num_channels=[30]*7,
                 kernel_size=3,
                 repo_initialization=True,
                 dropout=0.1,
                 optimizer="Adam",
                 loss="mse",
                 lr=0.001,
                 metrics=["mse"],
                 seed=None,
                 distributed=False,
                 workers_per_node=1,
                 distributed_backend="ray"):
        """
        Build a TCN Forecast Model.

        TCN Forecast may fall into local optima. Please set repo_initialization
        to False to alleviate the issue. You can also change a random seed to
        work around.

        :param past_seq_len: Specify the history time steps (i.e. lookback).
        :param future_seq_len: Specify the output time steps (i.e. horizon).
        :param input_feature_num: Specify the feature dimension.
        :param output_feature_num: Specify the output dimension.
        :param num_channels: Specify the convolutional layer filter number in
               TCN's encoder. This value defaults to [30]*7.
        :param kernel_size: Specify convolutional layer filter height in TCN's
               encoder. This value defaults to 3.
        :param repo_initialization: if to use framework default initialization,
               True to use paper author's initialization and False to use the
               framework's default initialization. The value defaults to True.
        :param dropout: Specify the dropout close possibility (i.e. the close
               possibility to a neuron). This value defaults to 0.1.
        :param optimizer: Specify the optimizer used for training. This value
               defaults to "Adam".
        :param loss: str or pytorch loss instance, Specify the loss function
               used for training. This value defaults to "mse". You can choose
               from "mse", "mae", "huber_loss" or any customized loss instance
               you want to use.
        :param lr: Specify the learning rate. This value defaults to 0.001.
        :param metrics: A list contains metrics for evaluating the quality of
               forecasting. You may only choose from "mse" and "mae" for a
               distributed forecaster. You may choose from "mse", "mae",
               "rmse", "r2", "mape", "smape" or a callable function for a
               non-distributed forecaster. If callable function, it signature
               should be func(y_true, y_pred), where y_true and y_pred are numpy
               ndarray.
        :param seed: int, random seed for training. This value defaults to None.
        :param distributed: bool, if init the forecaster in a distributed
               fashion. If True, the internal model will use an Orca Estimator.
               If False, the internal model will use a pytorch model. The value
               defaults to False.
        :param workers_per_node: int, the number of worker you want to use.
               The value defaults to 1. The param is only effective when
               distributed is set to True.
        :param distributed_backend: str, select from "ray" or
               "horovod". The value defaults to "ray".
        """
        # config setting
        self.data_config = {
            "past_seq_len": past_seq_len,
            "future_seq_len": future_seq_len,
            "input_feature_num": input_feature_num,
            "output_feature_num": output_feature_num
        }
        self.model_config = {
            "num_channels": num_channels,
            "kernel_size": kernel_size,
            "repo_initialization": repo_initialization,
            "dropout": dropout
        }
        self.loss_config = {
            "loss": loss
        }
        self.optim_config = {
            "lr": lr,
            "optim": optimizer
        }

        # model creator settings
        self.model_creator = model_creator
        self.optimizer_creator = optimizer_creator
        if isinstance(loss, str):
            self.loss_creator = loss_creator
        else:
            def customized_loss_creator(config):
                return config["loss"]
            self.loss_creator = customized_loss_creator

        # distributed settings
        self.distributed = distributed
        self.distributed_backend = distributed_backend
        self.workers_per_node = workers_per_node

        # other settings
        self.lr = lr
        self.metrics = metrics
        self.seed = seed

        # nano setting
        current_num_threads = torch.get_num_threads()
        if current_num_threads >= 24:
            self.num_processes = max(1, current_num_threads//8)  # 8 is a magic num
        else:
            self.num_processes = 1
        self.use_ipex = False  # TCN has worse performance on ipex
        self.onnx_available = True
        self.quantize_available = True
        self.checkpoint_callback = True
        self.use_hpo = True

        super().__init__()

    @staticmethod
    def from_tsdataset(tsdataset, past_seq_len=None, future_seq_len=None, **kwargs):
        '''
        Build a TCN Forecaster Model.

        :param tsdataset: A bigdl.chronos.data.tsdataset.TSDataset instance.
        :param past_seq_len: Specify the history time steps (i.e. lookback).
               No need to specify past_seq_len if tsdataset has called
               the 'roll' method.
        :param future_seq_len: Specify the output time steps (i.e. horizon).
               Same as past_seq_len if tsdataset has called
               the 'roll' method.
        :param kwargs: Specify parameters of Forecaster,
                e.g. loss and optimizer, etc.

        return: A TCN Forecaster Model.
        '''
        if tsdataset.numpy_x is not None:
            past_seq_len = tsdataset.numpy_x.shape[1]
            future_seq_len = tsdataset.numpy_y.shape[1]
        # TODO Support specifying 'auto' as past_seq_len.
        if all([past_seq_len is None, future_seq_len is None, tsdataset.numpy_x is None]):
            from bigdl.nano.utils.log4Error import invalidInputError
            invalidInputError(False,
                              "Need to specify 'past_seq_len' and 'future_seq_len' for"
                              " from_dataset or call the 'roll' method of dataset.")
        output_feature_num = len(tsdataset.target_col)
        input_feature_num = output_feature_num + len(tsdataset.feature_col)
        return TCNForecaster(past_seq_len=past_seq_len,
                             future_seq_len=future_seq_len,
                             input_feature_num=input_feature_num,
                             output_feature_num=output_feature_num,
                             **kwargs)
