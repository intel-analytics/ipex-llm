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
from bigdl.chronos.model.VanillaLSTM_pytorch import model_creator, optimizer_creator, loss_creator


class LSTMForecaster(BasePytorchForecaster):
    """
    Example:
        >>> #The dataset is split into x_train, x_val, x_test, y_train, y_val, y_test
        >>> # 1. Initialize Forecaster directly
        >>> forecaster = LSTMForecaster(past_seq_len=24,
                                        input_feature_num=2,
                                        output_feature_num=2,
                                        ...)
        >>>
        >>> # 2. Initialize Forecaster from from_tsdataset
        >>> forecaster = LSTMForecaster.from_tsdataset(tsdata, **kwargs)
        >>> forecaster.fit(tsdata, epochs=2, ...)
        >>> forecaster.to_local()  # if you set distributed=True
        >>> test_pred = forecaster.predict(x_test)
        >>> test_eval = forecaster.evaluate((x_test, y_test))
        >>> forecaster.save({ckpt_name})
        >>> forecaster.load({ckpt_name})
    """

    def __init__(self,
                 past_seq_len,
                 input_feature_num,
                 output_feature_num,
                 hidden_dim=32,
                 layer_num=1,
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
        Build a LSTM Forecast Model.

        :param past_seq_len: Specify the history time steps (i.e. lookback).
        :param input_feature_num: Specify the feature dimension.
        :param output_feature_num: Specify the output dimension.
        :param hidden_dim: int or list, Specify the hidden dim of each lstm layer.
               The value defaults to 32.
        :param layer_num: Specify the number of lstm layer to be used. The value
               defaults to 1.
        :param dropout: int or list, Specify the dropout close possibility
               (i.e. the close possibility to a neuron). This value defaults to 0.1.
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
            "future_seq_len": 1,  # lstm model only supports 1 step prediction
            "input_feature_num": input_feature_num,
            "output_feature_num": output_feature_num
        }
        self.model_config = {
            "hidden_dim": hidden_dim,
            "layer_num": layer_num,
            "dropout": dropout,
            "seed": seed
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
        self.remote_distributed_backend = distributed_backend
        self.local_distributed_backend = "subprocess"
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
        self.use_ipex = False
        self.onnx_available = True
        self.quantize_available = True
        self.checkpoint_callback = True
        self.use_hpo = True

        super().__init__()

    @classmethod
    def from_tsdataset(cls, tsdataset, past_seq_len=None, **kwargs):
        '''
        Build a LSTM Forecaster Model.

        :param tsdataset: Train tsdataset, a bigdl.chronos.data.tsdataset.TSDataset instance.
        :param past_seq_len: Specify the history time steps (i.e. lookback).
               Do not specify the 'past_seq_len' if your tsdataset has called
               the 'TSDataset.roll' method or 'TSDataset.to_torch_data_loader'.
        :param kwargs: Specify parameters of Forecaster,
               e.g. loss and optimizer, etc. More info, please refer to
               LSTMForecaster.__init__ methods.

        :return: A LSTM Forecaster Model.
        '''
        from bigdl.chronos.data.tsdataset import TSDataset
        from bigdl.nano.utils.log4Error import invalidInputError

        invalidInputError(isinstance(tsdataset, TSDataset),
                          f"We only supports input a TSDataset, but get{type(tsdataset)}.")

        def check_time_steps(tsdataset, past_seq_len):
            if tsdataset.lookback is not None and past_seq_len is not None:
                return tsdataset.lookback == past_seq_len
            return True

        invalidInputError(not tsdataset._has_generate_agg_feature,
                          "We will add su`pport for 'gen_rolling_feature' method later.")

        if tsdataset.lookback is not None:  # calling roll or to_torch_data_loader
            past_seq_len = tsdataset.lookback
            output_feature_num = len(tsdataset.roll_target)
            input_feature_num = len(tsdataset.roll_feature) + output_feature_num
        elif past_seq_len is not None:  # initialize only
            past_seq_len = past_seq_len if isinstance(past_seq_len, int)\
                else tsdataset.get_cycle_length()
            output_feature_num = len(tsdataset.target_col)
            input_feature_num = len(tsdataset.feature_col) + output_feature_num
        else:
            invalidInputError(False,
                              "Forecaster needs 'past_seq_len' to specify "
                              "the history time step of training.")

        invalidInputError(check_time_steps(tsdataset, past_seq_len),
                          "tsdataset already has history time steps and "
                          "differs from the given past_seq_len "
                          f"Expected past_seq_len to be {tsdataset.lookback}, "
                          f"but found {past_seq_len}.",
                          fixMsg="Do not specify past_seq_len "
                          "or call tsdataset.roll method again and specify time step.")

        if tsdataset.id_sensitive:
            _id_list_len = len(tsdataset._id_list)
            input_feature_num *= _id_list_len
            output_feature_num *= _id_list_len

        return cls(past_seq_len=past_seq_len,
                   input_feature_num=input_feature_num,
                   output_feature_num=output_feature_num,
                   **kwargs)
