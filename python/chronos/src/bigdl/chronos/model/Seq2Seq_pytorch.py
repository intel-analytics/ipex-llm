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
import torch.nn as nn

from .utils import PYTORCH_REGRESSION_LOSS_MAP
import numpy as np
from pytorch_lightning import seed_everything
from bigdl.chronos.pytorch.model_wrapper.normalization import NormalizeTSModel
from bigdl.chronos.pytorch.model_wrapper.decomposition import DecompositionTSModel


class LSTMSeq2Seq(nn.Module):
    def __init__(self,
                 input_feature_num,
                 future_seq_len,
                 output_feature_num,
                 lstm_hidden_dim=128,
                 lstm_layer_num=2,
                 dropout=0.25,
                 teacher_forcing=False,
                 seed=None):
        super(LSTMSeq2Seq, self).__init__()
        seed_everything(seed, workers=True)
        self.lstm_encoder = nn.LSTM(input_size=input_feature_num,
                                    hidden_size=lstm_hidden_dim,
                                    num_layers=lstm_layer_num,
                                    dropout=dropout,
                                    batch_first=True)
        self.lstm_decoder = nn.LSTM(input_size=output_feature_num,
                                    hidden_size=lstm_hidden_dim,
                                    num_layers=lstm_layer_num,
                                    dropout=dropout,
                                    batch_first=True)
        self.fc = nn.Linear(in_features=lstm_hidden_dim, out_features=output_feature_num)
        self.future_seq_len = future_seq_len
        self.output_feature_num = output_feature_num
        self.teacher_forcing = teacher_forcing

    def forward(self, input_seq, target_seq=None):
        x, (hidden, cell) = self.lstm_encoder(input_seq)
        # input feature order should have target dimensions in the first
        decoder_input = input_seq[:, -1, :self.output_feature_num]
        decoder_input = decoder_input.unsqueeze(1)
        decoder_output = []
        for i in range(self.future_seq_len):
            decoder_output_step, (hidden, cell) = self.lstm_decoder(decoder_input, (hidden, cell))
            out_step = self.fc(decoder_output_step)
            decoder_output.append(out_step)
            if not self.teacher_forcing or target_seq is None:
                # no teaching force
                decoder_input = out_step
            else:
                # with teaching force
                decoder_input = target_seq[:, i:i+1, :]
        decoder_output = torch.cat(decoder_output, dim=1)
        return decoder_output


def model_creator(config):
    model = LSTMSeq2Seq(input_feature_num=config["input_feature_num"],
                        output_feature_num=config["output_feature_num"],
                        future_seq_len=config["future_seq_len"],
                        lstm_hidden_dim=config.get("lstm_hidden_dim", 128),
                        lstm_layer_num=config.get("lstm_layer_num", 2),
                        dropout=config.get("dropout", 0.25),
                        teacher_forcing=config.get("teacher_forcing", False),
                        seed=config.get("seed", None))
    if config.get("normalization", False):
        model = NormalizeTSModel(model, config["output_feature_num"])
    decomposition_kernel_size = config.get("decomposition_kernel_size", 0)
    if decomposition_kernel_size > 1:
        model_copy = LSTMSeq2Seq(input_feature_num=config["input_feature_num"],
                                 output_feature_num=config["output_feature_num"],
                                 future_seq_len=config["future_seq_len"],
                                 lstm_hidden_dim=config.get("lstm_hidden_dim", 128),
                                 lstm_layer_num=config.get("lstm_layer_num", 2),
                                 dropout=config.get("dropout", 0.25),
                                 teacher_forcing=config.get("teacher_forcing", False),
                                 seed=config.get("seed", None))
        if config.get("normalization", False):
            model_copy = NormalizeTSModel(model_copy, config["output_feature_num"])
        model = DecompositionTSModel((model, model_copy), decomposition_kernel_size)
    return model


def optimizer_creator(model, config):
    return getattr(torch.optim, config.get("optim", "Adam"))(model.parameters(),
                                                             lr=config.get("lr", 0.001))


def loss_creator(config):
    loss_name = config.get("loss", "mse")
    if loss_name in PYTORCH_REGRESSION_LOSS_MAP:
        loss_name = PYTORCH_REGRESSION_LOSS_MAP[loss_name]
    else:
        from bigdl.nano.utils.common import invalidInputError
        invalidInputError(False,
                          f"Got '{loss_name}' for loss name, "
                          "where 'mse', 'mae' or 'huber_loss' is expected")
    return getattr(torch.nn, loss_name)()


try:
    from bigdl.orca.automl.model.base_pytorch_model import PytorchBaseModel

    class Seq2SeqPytorch(PytorchBaseModel):
        def __init__(self, check_optional_config=False):
            super().__init__(model_creator=model_creator,
                             optimizer_creator=optimizer_creator,
                             loss_creator=loss_creator,
                             check_optional_config=check_optional_config)

        def _input_check(self, x, y):
            from bigdl.nano.utils.common import invalidInputError
            if len(x.shape) < 3:
                invalidInputError(False,
                                  f"Invalid data x with {len(x.shape)} "
                                  "dim where 3 dim is required.")
            if len(y.shape) < 3:
                invalidInputError(False,
                                  f"Invalid data y with {len(y.shape)} dim "
                                  "where 3 dim is required.")
            if y.shape[-1] > x.shape[-1]:
                invalidInputError(False,
                                  "output dim should not larger than input dim "
                                  f"while we get {y.shape[-1]} > {x.shape[-1]}.")

        def _forward(self, x, y):
            self._input_check(x, y)
            return self.model(x, y)

        def _get_required_parameters(self):
            return {
                "input_feature_num",
                "future_seq_len",
                "output_feature_num"
            }

        def _get_optional_parameters(self):
            return {
                "lstm_hidden_dim",
                "lstm_layer_num",
                "teacher_forcing"
            } | super()._get_optional_parameters()
except ImportError:
    pass
