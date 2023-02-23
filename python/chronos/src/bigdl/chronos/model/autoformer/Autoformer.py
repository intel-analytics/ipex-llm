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

# MIT License

# Copyright (c) 2021 THUML @ Tsinghua University

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# code adapted from https://github.com/thuml/Autoformer


import torch
import torch.nn as nn
from .layers.Embed import DataEmbedding_wo_pos
from .layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer,\
    DecoderLayer, my_Layernorm, series_decomp
import torch.optim as optim
import pytorch_lightning as pl

from collections import namedtuple
from ..utils import PYTORCH_REGRESSION_LOSS_MAP


class AutoFormer(pl.LightningModule):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super().__init__()
        kwargs = {k: getattr(configs, k) for k in configs._fields}
        self.save_hyperparameters(kwargs)
        pl.seed_everything(configs.seed, workers=True)
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.optim = configs.optim
        self.lr = configs.lr
        self.lr_scheduler_milestones = configs.lr_scheduler_milestones
        self.loss = loss_creator(configs.loss)
        self.c_out = configs.c_out

        # Decomp
        # change kernei_size to odd
        kernel_size = int(2 * (configs.moving_avg // 2)) + 1
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed,
                                                  configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed,
                                                  configs.freq, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=kernel_size,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=kernel_size,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_enc.shape[0], self.pred_len, x_enc.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask,
                                                 cross_mask=dec_enc_mask, trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

    def training_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = map(lambda x: x.float(), batch)
        outputs = self(batch_x, batch_x_mark, batch_y, batch_y_mark)

        outputs = outputs[:, -self.pred_len:, -self.c_out:]
        batch_y = batch_y[:, -self.pred_len:, -self.c_out:]
        return self.loss(outputs, batch_y)

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = map(lambda x: x.float(), batch)
        outputs = self(batch_x, batch_x_mark, batch_y, batch_y_mark)

        outputs = outputs[:, -self.pred_len:, -self.c_out:]
        batch_y = batch_y[:, -self.pred_len:, -self.c_out:]
        self.log("val_loss", self.loss(outputs, batch_y))

    def predict_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = map(lambda x: x.float(), batch)
        outputs = self(batch_x, batch_x_mark, batch_y, batch_y_mark)

        outputs = outputs[:, -self.pred_len:, -self.c_out:]
        return outputs

    def configure_optimizers(self):
        optimizer = getattr(optim, self.optim)(self.parameters(), lr=self.lr)
        if self.lr_scheduler_milestones is not None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, gamma=0.5, verbose=True,
                milestones=self.lr_scheduler_milestones)
            return [optimizer], [scheduler]
        else:
            return optimizer


def model_creator(config):
    args = _transform_config_to_namedtuple(config)
    return AutoFormer(args)


def loss_creator(loss_name):
    if loss_name in PYTORCH_REGRESSION_LOSS_MAP:
        loss_name = PYTORCH_REGRESSION_LOSS_MAP[loss_name]
    else:
        from bigdl.nano.utils.common import invalidInputError
        invalidInputError(False,
                          f"Got '{loss_name}' for loss name, "
                          "where 'mse', 'mae' or 'huber_loss' is expected")
    return getattr(torch.nn, loss_name)()


def _transform_config_to_namedtuple(config):
    args = namedtuple("config", ['seq_len', 'label_len',
                                 'pred_len', 'output_attention',
                                 'moving_avg', 'enc_in',
                                 'd_model', 'embed',
                                 'freq', 'dropout',
                                 'dec_in', 'factor',
                                 'n_heads', 'd_ff',
                                 'activation', 'e_layers',
                                 'c_out', 'loss',
                                 'optim', 'lr',
                                 'lr_scheduler_milestones'])
    args.seq_len = config['seq_len']
    args.label_len = config['label_len']
    args.pred_len = config['pred_len']
    args.output_attention = config.get('output_attention', False)
    args.moving_avg = config.get('moving_avg', 25)
    args.enc_in = config['enc_in']
    args.d_model = config.get('d_model', 512)
    args.embed = config.get('embed', 'timeF')
    args.freq = config['freq']
    args.dropout = config.get('dropout', 0.05)
    args.dec_in = config['dec_in']
    args.factor = config.get('factor', 3)
    args.n_heads = config.get('n_heads', 8)
    args.d_ff = config.get('d_ff', 2048)
    args.activation = config.get('activation', 'gelu')
    args.e_layers = config.get('e_layers', 2)
    args.c_out = config['c_out']
    args.d_layers = config.get('d_layers', 1)
    args.loss = config.get("loss", "mse")
    args.optim = config.get("optim", "Adam")
    args.lr = config.get("lr", 0.0001)
    args.lr_scheduler_milestones = config.get("lr_scheduler_milestones", None)
    args.seed = config.get("seed", None)

    return args
