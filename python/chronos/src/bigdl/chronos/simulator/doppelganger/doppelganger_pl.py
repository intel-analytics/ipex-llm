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

# The Clear BSD License

# Copyright (c) 2019 Carnegie Mellon University
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification,
# are permitted (subject to the limitations in the disclaimer below) provided that
# the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#     * Neither the name of Carnegie Mellon University nor the names of its contributors
#       may be used to endorse or promote products derived from this software without
#       specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import os
import numpy as np
import sys
from collections import OrderedDict
import math

import torch
import torch.nn.functional as F
from torch import nn

from pytorch_lightning import LightningModule
from pytorch_lightning import seed_everything

from .doppelganger import DoppelGANger
from .network import RNNInitialStateType
from .loss import doppelganger_loss
from .util import gen_attribute_input_noise, gen_feature_input_noise,\
    gen_feature_input_data_free, renormalize_per_sample


class DoppelGANger_pl(LightningModule):
    def __init__(self,
                 data_feature_outputs,
                 data_attribute_outputs,
                 L_max,
                 num_real_attribute,
                 sample_len=10,
                 discriminator_num_layers=5,
                 discriminator_num_units=200,
                 attr_discriminator_num_layers=5,
                 attr_discriminator_num_units=200,
                 attribute_num_units=100,
                 attribute_num_layers=3,
                 feature_num_units=100,
                 feature_num_layers=1,
                 attribute_input_noise_dim=5,
                 addi_attribute_input_noise_dim=5,
                 d_gp_coe=10,
                 attr_d_gp_coe=10,
                 g_attr_d_coe=1,
                 d_lr=0.001,
                 attr_d_lr=0.001,
                 g_lr=0.001,
                 g_rounds=1,
                 d_rounds=1,
                 **kwargs):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters("discriminator_num_layers",
                                  "discriminator_num_units",
                                  "attr_discriminator_num_layers",
                                  "attr_discriminator_num_units",
                                  "attribute_num_units",
                                  "attribute_num_layers",
                                  "feature_num_units",
                                  "feature_num_layers",
                                  "attribute_input_noise_dim",
                                  "addi_attribute_input_noise_dim",
                                  "d_gp_coe",
                                  "attr_d_gp_coe",
                                  "g_attr_d_coe",
                                  "d_lr",
                                  "attr_d_lr",
                                  "g_lr",
                                  "g_rounds",
                                  "d_rounds",
                                  "L_max",
                                  "sample_len",
                                  "num_real_attribute")
        self.g_rounds = g_rounds
        self.d_rounds = d_rounds
        self.sample_len = sample_len
        self.L_max = L_max
        self.data_feature_outputs = data_feature_outputs
        self.data_attribute_outputs = data_attribute_outputs

        self.length = self.L_max // self.sample_len
        self.real_attribute_mask = ([True] * num_real_attribute +
                                    [False] * (len(data_attribute_outputs)-num_real_attribute))
        self.gen_flag_dims = []
        dim = 0
        from bigdl.nano.utils.common import invalidInputError
        for output in self.data_feature_outputs:
            if output.is_gen_flag:
                if output.dim != 2:
                    invalidInputError(False,
                                      "gen flag output's dim should be 2")
                self.gen_flag_dims = [dim, dim + 1]
                break
            dim += output.dim
        if len(self.gen_flag_dims) == 0:
            invalidInputError(False, "gen flag not found")

        # model init
        self.model =\
            DoppelGANger(
                data_feature_outputs=self.data_feature_outputs,
                data_attribute_outputs=self.data_attribute_outputs,
                real_attribute_mask=self.real_attribute_mask,
                sample_len=self.sample_len,
                L_max=self.L_max,
                num_packing=1,  # any num other than 1 will be supported later
                discriminator_num_layers=self.hparams.discriminator_num_layers,
                discriminator_num_units=self.hparams.discriminator_num_units,
                attr_discriminator_num_layers=self.hparams.attr_discriminator_num_layers,
                attr_discriminator_num_units=self.hparams.attr_discriminator_num_units,
                attribute_num_units=self.hparams.attribute_num_units,
                attribute_num_layers=self.hparams.attribute_num_layers,
                feature_num_units=self.hparams.feature_num_units,
                feature_num_layers=self.hparams.feature_num_layers,
                attribute_input_noise_dim=self.hparams.attribute_input_noise_dim,
                addi_attribute_input_noise_dim=self.hparams.addi_attribute_input_noise_dim,
                initial_state=RNNInitialStateType.RANDOM)  # currently we fix this value

    def forward(self,
                data_feature,
                real_attribute_input_noise,
                addi_attribute_input_noise,
                feature_input_noise,
                data_attribute):
        return self.model([data_feature],
                          [real_attribute_input_noise],
                          [addi_attribute_input_noise],
                          [feature_input_noise],
                          [data_attribute])

    def training_step(self, batch, batch_idx):
        # data preparation
        data_feature, data_attribute = batch
        optimizer_d, optimizer_attr_d, optimizer_g = self.optimizers()

        # generate noise input
        real_attribute_input_noise = gen_attribute_input_noise(data_feature.shape[0])
        addi_attribute_input_noise = gen_attribute_input_noise(data_feature.shape[0])
        feature_input_noise = gen_feature_input_noise(data_feature.shape[0], self.length)
        real_attribute_input_noise = torch.from_numpy(real_attribute_input_noise).float()
        addi_attribute_input_noise = torch.from_numpy(addi_attribute_input_noise).float()
        feature_input_noise = torch.from_numpy(feature_input_noise).float()

        # g backward
        # open the generator grad since we need to update the weights in g
        for p in self.model.generator.parameters():
            p.requires_grad = True
        for i in range(self.g_rounds):
            d_fake, attr_d_fake,\
                d_real, attr_d_real = self(data_feature,
                                           real_attribute_input_noise,
                                           addi_attribute_input_noise,
                                           feature_input_noise,
                                           data_attribute)
            g_loss, _, _ =\
                doppelganger_loss(d_fake, attr_d_fake, d_real, attr_d_real)
            optimizer_g.zero_grad()
            self.manual_backward(g_loss)
            optimizer_g.step()

        # d backward
        # close the generator grad since we only need to update the weights in d
        for p in self.model.generator.parameters():
            p.requires_grad = False
        for i in range(self.d_rounds):
            d_fake, attr_d_fake,\
                d_real, attr_d_real = self(data_feature,
                                           real_attribute_input_noise,
                                           addi_attribute_input_noise,
                                           feature_input_noise,
                                           data_attribute)
            _, d_loss, attr_d_loss =\
                doppelganger_loss(d_fake, attr_d_fake, d_real, attr_d_real,
                                  g_attr_d_coe=self.hparams.g_attr_d_coe,
                                  gradient_penalty=True,
                                  discriminator=self.model.discriminator,
                                  attr_discriminator=self.model.attr_discriminator,
                                  g_output_feature_train_tf=self.model.g_feature_train,
                                  g_output_attribute_train_tf=self.model.g_attribute_train,
                                  real_feature_pl=self.model.real_feature_pl,
                                  real_attribute_pl=self.model.real_attribute_pl,
                                  d_gp_coe=self.hparams.d_gp_coe,
                                  attr_d_gp_coe=self.hparams.attr_d_gp_coe)
            optimizer_d.zero_grad()
            optimizer_attr_d.zero_grad()
            self.manual_backward(d_loss)
            self.manual_backward(attr_d_loss)
            optimizer_d.step()
            optimizer_attr_d.step()

            # log tqdm
            self.log("g_loss", g_loss.item(),
                     on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("d_loss", d_loss.item(),
                     on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("attr_d_loss", attr_d_loss.item(),
                     on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer_d = torch.optim.Adam(self.model.discriminator.parameters(),
                                       lr=self.hparams.d_lr, betas=(0.5, 0.999))
        optimizer_attr_d = torch.optim.Adam(self.model.attr_discriminator.parameters(),
                                            lr=self.hparams.attr_d_lr, betas=(0.5, 0.999))
        optimizer_g = torch.optim.Adam(self.model.generator.parameters(),
                                       lr=self.hparams.g_lr, betas=(0.5, 0.999))
        return optimizer_d, optimizer_attr_d, optimizer_g

    def sample_from(self,
                    real_attribute_input_noise,
                    addi_attribute_input_noise,
                    feature_input_noise,
                    feature_input_data,
                    batch_size=32):
        features, attributes, gen_flags, lengths\
            = self.model.sample_from(real_attribute_input_noise,
                                     addi_attribute_input_noise,
                                     feature_input_noise,
                                     feature_input_data,
                                     self.gen_flag_dims,
                                     batch_size=batch_size)
        return features, attributes, gen_flags, lengths
