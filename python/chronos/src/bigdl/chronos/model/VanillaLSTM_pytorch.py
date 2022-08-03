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
from pytorch_lightning import seed_everything


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num, dropout, output_dim, seed):
        super(LSTMModel, self).__init__()
        seed_everything(seed, workers=True)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.layer_num = layer_num
        lstm_list = []
        for layer in range(self.layer_num):
            lstm_list.append(nn.LSTM(input_dim, self.hidden_dim[layer],
                                     1, dropout=self.dropout[layer], batch_first=True))
            input_dim = self.hidden_dim[layer]
        self.lstm = nn.ModuleList(lstm_list)
        self.fc = nn.Linear(self.hidden_dim[-1], output_dim)
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, input_seq):
        lstm_out = input_seq
        for layer in range(self.layer_num):
            lstm_out, _ = self.lstm[layer](lstm_out)
        out = self.fc(lstm_out[:, -1, :])
        out = out.view(out.shape[0], 1, out.shape[1])
        return out


def model_creator(config):
    hidden_dim = config.get("hidden_dim", 32)
    dropout = config.get("dropout", 0.2)
    layer_num = config.get("layer_num", 2)
    from bigdl.nano.utils.log4Error import invalidInputError
    if isinstance(hidden_dim, list):
        invalidInputError(len(hidden_dim) == layer_num,
                          "length of hidden_dim should be equal to layer_num")
    if isinstance(dropout, list):
        invalidInputError(len(dropout) == layer_num,
                          "length of dropout should be equal to layer_num")
    if isinstance(hidden_dim, int):
        hidden_dim = [hidden_dim]*layer_num
    if isinstance(dropout, (float, int)):
        dropout = [dropout]*layer_num
    return LSTMModel(input_dim=config["input_feature_num"],
                     hidden_dim=hidden_dim,
                     layer_num=layer_num,
                     dropout=dropout,
                     output_dim=config["output_feature_num"],
                     seed=config.get("seed", None))


def optimizer_creator(model, config):
    return getattr(torch.optim, config.get("optim", "Adam"))(model.parameters(),
                                                             lr=config.get("lr", 0.001))


def loss_creator(config):
    return nn.MSELoss()


try:
    from bigdl.orca.automl.model.base_pytorch_model import PytorchBaseModel

    class VanillaLSTMPytorch(PytorchBaseModel):

        def __init__(self, check_optional_config=False):
            """
            Constructor of Vanilla LSTM model
            """
            super().__init__(model_creator=model_creator,
                             optimizer_creator=optimizer_creator,
                             loss_creator=loss_creator,
                             check_optional_config=check_optional_config)

        def _get_required_parameters(self):
            return {
                "input_feature_num",
                "output_feature_num"
            }

        def _get_optional_parameters(self):
            return {
                'hidden_dim',
                'layer_num',
                'dropout',
            } | super()._get_optional_parameters()
except ImportError:
    pass
