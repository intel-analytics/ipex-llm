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
from bigdl.chronos.model.autoformer import model_creator, loss_creator
from torch.utils.data import TensorDataset, DataLoader
from bigdl.chronos.model.autoformer.Autoformer import AutoFormer, _transform_config_to_namedtuple
from bigdl.nano.utils.log4Error import invalidInputError, invalidOperationError
from bigdl.chronos.forecaster.utils import check_transformer_data
from bigdl.chronos.pytorch import TSTrainer as Trainer
from bigdl.nano.automl.hpo.space import Space
from bigdl.chronos.forecaster.utils_hpo import GenericTSTransformerLightningModule, \
    _config_has_search_space

from .utils_hpo import _format_metric_str
import warnings


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
                 e_layers=2,
                 d_layers=1,
                 optimizer="Adam",
                 loss="mse",
                 lr=0.0001,
                 lr_scheduler_milestones=[3, 4, 5, 6, 7, 8, 9, 10],
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
        :param label_len: Start token length of AutoFormer decoder.
        :param freq: Freq for time features encoding. You may choose from "s",
               "t","h","d","w","m" for second, minute, hour, day, week or month.
        :param optimizer: Specify the optimizer used for training. This value
               defaults to "Adam".
        :param loss: str or pytorch loss instance, Specify the loss function
               used for training. This value defaults to "mse". You can choose
               from "mse", "mae", "huber_loss" or any customized loss instance
               you want to use.
        :param lr: Specify the learning rate. This value defaults to 0.001.
        :param lr_scheduler_milestones: Specify the milestones parameters in
               torch.optim.lr_scheduler.MultiStepLR.This value defaults to
               [3, 4, 5, 6, 7, 8, 9, 10]. If you don't want to use scheduler,
               set this parameter to None to disbale lr_scheduler.
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
            "output_feature_num": output_feature_num,
            "label_len": label_len
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
            "e_layers": e_layers,
            "c_out": output_feature_num,
            "d_layers": d_layers,
            "seed": seed,
        }
        self.loss_config = {
            "loss": loss
        }
        self.optim_config = {
            "lr": lr,
            "optim": optimizer,
            "lr_scheduler_milestones": lr_scheduler_milestones,
        }

        self.model_config.update(self.loss_config)
        self.model_config.update(self.optim_config)

        self.metrics = metrics

        self.distributed = distributed
        self.checkpoint_callback = True
        # seed setting
        if not isinstance(seed, Space):
            from pytorch_lightning import seed_everything
            seed_everything(seed=seed, workers=True)

        # disable multi-process training for now.
        # TODO: enable it in future.
        self.num_processes = 1
        self.use_ipex = False
        self.onnx_available = False
        self.quantize_available = False
        self.use_amp = False
        self.use_hpo = True

        has_space = _config_has_search_space(
            config={**self.model_config, **self.optim_config,
                    **self.loss_config, **self.data_config})

        if not has_space:
            if self.use_hpo:
                warnings.warn("HPO is enabled but no spaces is specified, so disable HPO.")
            self.use_hpo = False
            self.internal = model_creator(self.model_config)

        self.model_creator = model_creator
        self.loss_creator = loss_creator

    def _build_automodel(self, data, validation_data=None, batch_size=32, epochs=1):
        """Build a Generic Model using config parameters."""
        merged_config = {**self.model_config, **self.optim_config,
                         **self.loss_config, **self.data_config}

        model_config_keys = list(self.model_config.keys())
        data_config_keys = list(self.data_config.keys())
        optim_config_keys = list(self.optim_config.keys())
        loss_config_keys = list(self.loss_config.keys())

        return GenericTSTransformerLightningModule(
            model_creator=self.model_creator,
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
             target_metric='mse',
             direction="minimize",
             n_trials=2,
             n_parallels=1,
             epochs=1,
             batch_size=32,
             **kwargs):
        """
        Search the hyper parameter.

        :param data: train data, as numpy ndarray tuple (x, y, x_enc, y_enc)
        :param validation_data: validation data, as numpy ndarray tuple (x, y, x_enc, y_enc)
        :param target_metric: the target metric to optimize,
               a string or an instance of torchmetrics.metric.Metric, default to 'mse'.
        :param direction: in which direction to optimize the target metric,
               "maximize" - larger the better
               "minimize" - smaller the better
               default to "minimize".
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
            check_transformer_data(data[0], data[1], data[2], data[3], self.data_config)
            if validation_data and isinstance(validation_data, tuple):
                check_transformer_data(validation_data[0], validation_data[1],
                                       validation_data[2], validation_data[3], self.data_config)
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

        self.trainer = Trainer(logger=False, max_epochs=epochs,
                               checkpoint_callback=self.checkpoint_callback,
                               num_processes=self.num_processes, use_ipex=self.use_ipex,
                               use_hpo=True,
                               callbacks=[ResetCallback()] if self.num_processes == 1 else None)

        # run hyper parameter search
        self.internal = self.trainer.search(
            self.tune_internal,
            n_trials=n_trials,
            target_metric=formated_target_metric,
            direction=direction,
            n_parallels=n_parallels,
            **kwargs)

        # reset train and validation datasets
        self.trainer.reset_train_val_dataloaders(self.internal)

    def search_summary(self):
        # add tuning check
        invalidOperationError(self.use_hpo, "No search summary when HPO is disabled.")
        return self.trainer.search_summary()

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

        from bigdl.chronos.pytorch import TSTrainer as Trainer
        # Trainer init and fitting
        if not self.use_hpo:
            self.trainer = Trainer(logger=False, max_epochs=epochs,
                                   checkpoint_callback=self.checkpoint_callback, num_processes=1,
                                   use_ipex=self.use_ipex, distributed_backend="spawn")
        else:
            # check whether the user called the tune function
            invalidOperationError(hasattr(self, "trainer"), "There is no trainer, and you "
                                  "should call .tune() before .fit()")

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


def _str2metric(metric):
    # map metric str to function
    if isinstance(metric, str):
        from bigdl.chronos.metric.forecast_metrics import TORCHMETRICS_REGRESSION_MAP
        metric = TORCHMETRICS_REGRESSION_MAP[metric]
    return metric
