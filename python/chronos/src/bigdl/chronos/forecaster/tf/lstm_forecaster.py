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

from bigdl.chronos.forecaster.tf.base_forecaster import BaseTF2Forecaster
from bigdl.chronos.model.tf2.VanillaLSTM_keras import model_creator, LSTMModel


class LSTMForecaster(BaseTF2Forecaster):
    """
    Example:
        >>> #The dataset is split into x_train, x_val, x_test, y_train, y_val, y_test
        >>> forecaster = LSTMForecaster(past_seq_len=24,
                                        input_feature_num=2,
                                        output_feature_num=2,
                                        ...)
        >>> forecaster.fit((x_train, y_train))
        >>> test_pred = forecaster.predict(x_test)
        >>> test_eval = forecaster.evaluate((x_test, y_test))
        >>> forecaster.save({ckpt_dir_name})
        >>> forecaster.load({ckpt_dir_name})
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
        :param loss: Str or a tf.keras.losses.Loss instance, specify the loss function
               used for training. This value defaults to "mse". You can choose
               from "mse", "mae" and "huber_loss" or any customized loss instance
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
               If False, the internal model will use a Keras model. The value
               defaults to False.
        :param workers_per_node: int, the number of worker you want to use.
               The value defaults to 1. The param is only effective when
               distributed is set to True.
        :param distributed_backend: str, select from "ray" or
               "horovod". The value defaults to "ray".
        """
        self.model_config = {
            "past_seq_len": past_seq_len,
            "future_seq_len": 1,  # lstm model only supports 1 step prediction
            "input_feature_num": input_feature_num,
            "output_feature_num": output_feature_num,
            "hidden_dim": hidden_dim,
            "layer_num": layer_num,
            "dropout": dropout,
            "loss": loss,
            "lr": lr,
            "optim": optimizer,
        }

        # model creator settings
        self.model_creator = model_creator
        self.custom_objects_config = {"LSTMModel": LSTMModel}

        # distributed settings
        # self.distributed = distributed
        # self.distributed_backend = distributed_backend
        # self.workers_per_node = workers_per_node
        from bigdl.nano.utils.log4Error import invalidInputError
        if distributed:
            invalidInputError(False,
                              "We will add distributed support in subsequent releases, "
                              "the feature is currently unavailable, "
                              "Please set distributed=False.")

        # other settings
        self.lr = lr
        self.metrics = metrics
        self.seed = seed

        # nano setting
        # current_num_threads = torch.get_num_threads()
        # self.num_processes = max(1, current_num_threads//8)  # 8 is a magic num
        # self.use_ipex = False
        # self.onnx_available = True
        # self.quantize_available = True
        # self.checkpoint_callback = False
        super(LSTMForecaster, self).__init__()
