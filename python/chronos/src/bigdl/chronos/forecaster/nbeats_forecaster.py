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
from bigdl.chronos.model.nbeats_pytorch import model_creator, loss_creator, optimizer_creator


class NBeatsForecaster(BasePytorchForecaster):
    """
    Example:
        >>> # 1. Initialize Forecaster directly
        >>> forecaster = NBeatForecaster(paste_seq_len=10,
                                         future_seq_len=1,
                                         stack_types=("generic", "generic"),
                                         ...)
        >>>
        >>> # 2. The from_tsdataset method can also initialize a NBeatForecaster.
        >>> forecaster.from_tsdataset(tsdata, **kwargs)
        >>> forecaster.fit(tsdata)
        >>> forecaster.to_local() # if you set distributed=True
    """

    def __init__(self,
                 past_seq_len,
                 future_seq_len,
                 stack_types=("generic", "generic"),
                 nb_blocks_per_stack=3,
                 thetas_dim=(4, 8),
                 share_weights_in_stack=False,
                 hidden_layer_units=256,
                 nb_harmonics=None,
                 optimizer="Adam",
                 loss="mse",
                 lr=0.001,
                 metrics=["mse"],
                 seed=None,
                 distributed=False,
                 workers_per_node=1,
                 distributed_backend="ray"):
        """
        Build a NBeats Forecaster Model.

        :param past_seq_len: Specify the history time steps (i.e. lookback).
        :param future_seq_len: Specify the output time steps (i.e. horizon).
        :param stack_types: Specifies the type of stack,
               including "generic", "trend", "seasnoality".
               This value defaults to ("generic", "generic").
               If set distributed=True, the second type should not be "generic",
               use "seasonality" or "trend", e.g. ("generic", "trend").
        :param nb_blocks_per_stack: Specify the number of blocks
               contained in each stack, This value defaults to 3.
        :param thetas_dim: Expansion Coefficients of Multilayer FC Networks.
               if type is "generic", Extended length factor, if type is "trend"
               then polynomial coefficients, if type is "seasonality"
               expressed as a change within each step.
        :param share_weights_in_stack: Share block weights for each stack.,
               This value defaults to False.
        :param hidden_layer_units: Number of fully connected layers with per block.
               This values defaults to 256.
        :param nb_harmonics: Only available in "seasonality" type,
               specifies the time step of backward, This value defaults is None.
        :param dropout: Specify the dropout close possibility
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
        # ("generic", "generic") not support orca distributed.
        if stack_types[-1] == "generic" and distributed:
            from bigdl.nano.utils.common import invalidInputError
            invalidInputError(False,
                              "Please set distributed=False or change the type "
                              "of 'stack_types' to 'trend', 'seasonality', "
                              "e.g. ('generic', 'seasonality').")

        self.data_config = {
            "past_seq_len": past_seq_len,
            "future_seq_len": future_seq_len,
            "input_feature_num": 1,  # nbeats only support input single feature.
            "output_feature_num": 1,
        }

        self.model_config = {
            "stack_types": stack_types,
            "nb_blocks_per_stack": nb_blocks_per_stack,
            "thetas_dim": thetas_dim,
            "share_weights_in_stack": share_weights_in_stack,
            "hidden_layer_units": hidden_layer_units,
            "nb_harmonics": nb_harmonics,
            "seed": seed,
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
        self.seed = seed
        self.metrics = metrics

        # nano settings
        current_num_threads = torch.get_num_threads()
        self.thread_num = current_num_threads
        self.optimized_model_thread_num = current_num_threads
        if current_num_threads >= 24:
            self.num_processes = max(1, current_num_threads//8)  # 8 is a magic num
        else:
            self.num_processes = 1
        self.use_ipex = False
        self.onnx_available = True
        self.quantize_available = True
        self.checkpoint_callback = True
        self.use_hpo = True
        self.optimized_model_output_tensor = True

        super().__init__()

    @classmethod
    def from_tsdataset(cls, tsdataset, past_seq_len=None, future_seq_len=None, **kwargs):
        """
        Build a NBeats Forecaster Model.

        :param tsdataset: Train tsdataset, a bigdl.chronos.data.tsdataset.TSDataset instance.
        :param past_seq_len: Specify the history time steps (i.e. lookback).
               Do not specify the 'past_seq_len' if your tsdataset has called
               the 'TSDataset.roll' method or 'TSDataset.to_torch_data_loader'.
        :param future_seq_len: Specify the output time steps (i.e. horizon).
               Do not specify the 'future_seq_len' if your tsdataset has called
               the 'TSDataset.roll' method or 'TSDataset.to_torch_data_loader'.
        :param kwargs: Specify parameters of Forecaster,
               e.g. loss and optimizer, etc. More info,
               please refer to NBeatsForecaster.__init__ methods.

        :return: A NBeats Forecaster Model.
        """
        from bigdl.chronos.data.tsdataset import TSDataset
        from bigdl.nano.utils.common import invalidInputError
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
        elif past_seq_len is not None and future_seq_len is not None:  # initialize only
            past_seq_len = past_seq_len if isinstance(past_seq_len, int)\
                else tsdataset.get_cycle_length()
            future_seq_len = future_seq_len if isinstance(future_seq_len, int) \
                else max(future_seq_len)
        else:
            invalidInputError(False,
                              "Forecaster requires 'past_seq_len' and 'future_seq_len' to specify "
                              "the history time step and output time step.")

        invalidInputError(check_time_steps(tsdataset, past_seq_len, future_seq_len),
                          "tsdataset already has historical time steps and "
                          "differs from the given past_seq_len and future_seq_len "
                          "Expected past_seq_len and future_seq_len to be "
                          f"{tsdataset.lookback, tsdataset.horizon}, "
                          f"but found {past_seq_len, future_seq_len}",
                          fixMsg="Do not specify past_seq_len and future seq_len "
                          "or call tsdataset.roll method again and specify time step")

        invalidInputError(not all([tsdataset.id_sensitive, len(tsdataset._id_list) > 1]),
                          "NBeats only supports univariate forecasting.")

        return cls(past_seq_len=past_seq_len,
                   future_seq_len=future_seq_len,
                   **kwargs)
