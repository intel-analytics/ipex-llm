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
from bigdl.chronos.forecaster.abstract import Forecaster
from bigdl.chronos.model.autoformer import model_creator
from torch.utils.data import TensorDataset, DataLoader
from bigdl.chronos.model.autoformer.Autoformer import AutoFormer, _transform_config_to_namedtuple
from bigdl.nano.utils.log4Error import invalidInputError
from bigdl.chronos.pytorch import TSTrainer as Trainer
import torch.nn as nn
import numpy as np
import os
from collections import namedtuple


class AutoformerForecaster(Forecaster):
    def __init__(self,
                 past_seq_len,
                 future_seq_len,
                 input_feature_num,
                 output_feature_num,
                 label_len,
                 freq,
                 output_attention=False,
                 moving_avg=25,
                 d_model=128,
                 embed='timeF',
                 dropout=0.05,
                 factor=3,
                 n_head=8,
                 d_ff=256,
                 activation='gelu',
                 e_layer=2,
                 d_layers=1,
                 optimizer="Adam",
                 loss="mse",
                 lr=0.0001,
                 metrics=["mse"],
                 seed=None,
                 distributed=False,
                 workers_per_node=1,
                 distributed_backend="ray"):

        """
        Build a AutoformerForecaster Forecast Model.

        :param past_seq_len: Specify the history time steps (i.e. lookback).
        :param future_seq_len: Specify the output time steps (i.e. horizon).
        :param input_feature_num: Specify the feature dimension.
        :param output_feature_num: Specify the output dimension.
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
        :param kwargs: other hyperparameter please refer to
               https://github.com/zhouhaoyi/Informer2020#usage
        """

        # config setting
        self.data_config = {
            "past_seq_len": past_seq_len,
            "future_seq_len": future_seq_len,
            "input_feature_num": input_feature_num,
            "output_feature_num": output_feature_num
        }
        self.model_config = {
            "seq_len": past_seq_len,
            "label_len": label_len,
            "pred_len": future_seq_len,
            "output_attention": output_attention,
            "moving_avg": moving_avg,
            "enc_in": input_feature_num,
            "d_model": d_model,
            "embed": embed,
            "freq": freq,
            "dropout": dropout,
            "dec_in": input_feature_num,
            "factor": factor,
            "n_head": n_head,
            "d_ff": d_ff,
            "activation": activation,
            "e_layer": e_layer,
            "c_out": output_feature_num,
            "d_layers": d_layers
        }
        self.loss_config = {
            "loss": loss
        }
        self.optim_config = {
            "lr": lr,
            "optim": optimizer
        }

        self.model_config.update(self.loss_config)
        self.model_config.update(self.optim_config)

        self.distributed = distributed
        self.seed = seed
        self.checkpoint_callback = True

        # disable multi-process training for now.
        # TODO: enable it in future.
        self.num_processes = 1
        self.use_ipex = False
        self.onnx_available = False
        self.quantize_available = False
        self.use_amp = False

        self.model_creator = model_creator
        self.internal = model_creator(self.model_config)

    def fit(self, data, epochs=1, batch_size=32):
        """
        Fit(Train) the forecaster.

        :param data: The data support following formats:

               | 1. numpy ndarrays: generate from `TSDataset.roll`,
                    be sure to set label_len > 0 and time_enc = True
               | 2. pytorch dataloader: generate from `TSDataset.to_torch_data_loader`,
                    be sure to set label_len > 0 and time_enc = True

        :param epochs: Number of epochs you want to train. The value defaults to 1.
        :param batch_size: Number of batch size you want to train. The value defaults to 32.
               if you input a pytorch dataloader for `data`, the batch_size will follow the
               batch_size setted in `data`.
        """
        # seed setting
        from pytorch_lightning import seed_everything
        seed_everything(seed=self.seed)

        # distributed is not supported.
        if self.distributed:
            invalidInputError(False, "distributed is not support in Autoformer")

        # transform a tuple to dataloader.
        if isinstance(data, tuple):
            data = DataLoader(TensorDataset(torch.from_numpy(data[0]),
                                            torch.from_numpy(data[1]),
                                            torch.from_numpy(data[2]),
                                            torch.from_numpy(data[3]),),
                              batch_size=batch_size,
                              shuffle=True)

        # Trainer init and fitting
        self.trainer = Trainer(logger=False, max_epochs=epochs,
                               checkpoint_callback=self.checkpoint_callback, num_processes=1,
                               use_ipex=self.use_ipex, distributed_backend="spawn")
        self.trainer.fit(self.internal, data)

    def predict(self, data, batch_size=32):
        """
        Predict using a trained forecaster.

        :param data: The data support following formats:

               | 1. numpy ndarrays: generate from `TSDataset.roll`,
                    be sure to set label_len > 0 and time_enc = True
               | 2. pytorch dataloader: generate from `TSDataset.to_torch_data_loader`,
                    be sure to set label_len > 0, time_enc = True and is_predict = True

        :param batch_size: predict batch size. The value will not affect predict
               result but will affect resources cost(e.g. memory and time).

        :return: A list of numpy ndarray
        """
        if self.distributed:
            invalidInputError(False, "distributed is not support in Autoformer")
        if isinstance(data, tuple):
            data = DataLoader(TensorDataset(torch.from_numpy(data[0]),
                                            torch.from_numpy(data[1]),
                                            torch.from_numpy(data[2]),
                                            torch.from_numpy(data[3]),),
                              batch_size=batch_size,
                              shuffle=False)

        return self.trainer.predict(self.internal, data)

    def evaluate(self, data, batch_size=32):
        """
        Predict using a trained forecaster.

        :param data: The data support following formats:

               | 1. numpy ndarrays: generate from `TSDataset.roll`,
                    be sure to set label_len > 0 and time_enc = True
               | 2. pytorch dataloader: generate from `TSDataset.to_torch_data_loader`,
                    be sure to set label_len > 0 and time_enc = True

        :param batch_size: predict batch size. The value will not affect predict
               result but will affect resources cost(e.g. memory and time).

        :return: A dict, currently returns the loss rather than metrics
        """
        # TODO: use metrics here
        if self.distributed:
            invalidInputError(False, "distributed is not support in Autoformer")
        if isinstance(data, tuple):
            data = DataLoader(TensorDataset(torch.from_numpy(data[0]),
                                            torch.from_numpy(data[1]),
                                            torch.from_numpy(data[2]),
                                            torch.from_numpy(data[3]),),
                              batch_size=batch_size,
                              shuffle=False)

        return self.trainer.validate(self.internal, data)

    def load(self, checkpoint_file):
        """
        restore the forecaster.

        :param checkpoint_file: The checkpoint file location you want to load the forecaster.
        """
        self.trainer = Trainer(logger=False, max_epochs=1,
                               checkpoint_callback=self.checkpoint_callback, num_processes=1,
                               use_ipex=self.use_ipex, distributed_backend="spawn")
        args = _transform_config_to_namedtuple(self.model_config)
        self.internal = AutoFormer.load_from_checkpoint(checkpoint_file, configs=args)

    def save(self, checkpoint_file):
        """
        save the forecaster.

        :param checkpoint_file: The checkpoint file location you want to load the forecaster.
        """
        self.trainer.save_checkpoint(checkpoint_file)
