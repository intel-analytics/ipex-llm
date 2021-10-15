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


import numpy as np
from tqdm import tqdm
import datetime
import os
import math
import sys
import torch
import torch.nn as nn
from .network import Discriminator, AttrDiscriminator, DoppelGANgerGenerator, RNNInitialStateType


class DoppelGANger(nn.Module):
    def __init__(self,
                 data_feature_outputs,
                 data_attribute_outputs,
                 real_attribute_mask,
                 sample_len,
                 L_max,
                 num_packing=1,
                 # discriminator parameters
                 discriminator_num_layers=5, discriminator_num_units=200,
                 # attr_discriminator parameters
                 attr_discriminator_num_layers=5, attr_discriminator_num_units=200,
                 # generator parameters
                 attribute_num_units=100, attribute_num_layers=3,
                 feature_num_units=100, feature_num_layers=2,
                 attribute_input_noise_dim=5, addi_attribute_input_noise_dim=5,
                 initial_state=RNNInitialStateType.RANDOM):
        '''
        :param data_feature_outputs: A list of Output objects, indicating the
            dimension, type, normalization of each feature
        :param data_attribute_outputs A list of Output objects, indicating the
            dimension, type, normalization of each attribute
        :param real_attribute_mask: List of True/False, the length equals the
            number of attributes. False if the attribute is (max-min)/2 or
            (max+min)/2, True otherwise
        :param num_packing: Packing degree in PacGAN (a method for solving mode
            collapse in NeurIPS 2018, see https://arxiv.org/abs/1712.04086), the
            value defaults to 1.
        '''
        super().__init__()
        self.data_feature_outputs = data_feature_outputs
        self.data_attribute_outputs = data_attribute_outputs
        self.num_packing = num_packing
        self.sample_len = sample_len
        self.real_attribute_mask = real_attribute_mask
        self.feature_out_dim = (np.sum([t.dim for t in data_feature_outputs]) *
                                self.sample_len)
        self.attribute_out_dim = np.sum([t.dim for t in data_attribute_outputs])

        self.generator\
            = DoppelGANgerGenerator(feed_back=False,
                                    # feed back mode has not been supported
                                    noise=True,
                                    feature_outputs=data_feature_outputs,
                                    attribute_outputs=data_attribute_outputs,
                                    real_attribute_mask=real_attribute_mask,
                                    sample_len=sample_len,
                                    attribute_num_units=attribute_num_units,
                                    attribute_num_layers=attribute_num_layers,
                                    feature_num_units=feature_num_units,
                                    feature_num_layers=feature_num_layers,
                                    attribute_input_noise_dim=attribute_input_noise_dim,
                                    addi_attribute_input_noise_dim=addi_attribute_input_noise_dim,
                                    attribute_dim=None,
                                    # known attribute feed-in has not been supported
                                    initial_state=initial_state,
                                    # only ZERO and RANDOM are supported
                                    initial_stddev=0.02)  # placehold without any usage
        self.discriminator\
            = Discriminator(input_size=(int(self.feature_out_dim*L_max /
                                            self.sample_len))*self.num_packing +
                            self.attribute_out_dim,
                            num_layers=discriminator_num_layers,
                            num_units=discriminator_num_units)
        self.attr_discriminator\
            = AttrDiscriminator(input_size=self.attribute_out_dim,
                                num_layers=attr_discriminator_num_layers,
                                num_units=attr_discriminator_num_units)

    def forward(self,
                data_feature,
                real_attribute_input_noise,
                addi_attribute_input_noise,
                feature_input_noise,
                data_attribute):
        # since we still not support self.num_packing
        self.data_feature = data_feature
        self.data_attribute = data_attribute

        if self.data_feature[0].shape[1] % self.sample_len != 0:
            raise Exception("length must be a multiple of sample_len")
        self.sample_time = int(self.data_feature[0].shape[1] / self.sample_len)
        self.sample_feature_dim = self.data_feature[0].shape[2]
        self.sample_attribute_dim = self.data_attribute[0].shape[1]

        self.batch_size = self.data_feature[0].shape[0]

        # generate training route (fake)
        self.g_output_feature_train_tf_l = []
        self.g_output_attribute_train_tf_l = []

        for i in range(self.num_packing):
            (g_output_feature_train_tf, g_output_attribute_train_tf,
             _, _, _) = \
                self.generator(real_attribute_input_noise[i],
                               addi_attribute_input_noise[i],
                               feature_input_noise[i],
                               data_feature[i])
            self.g_output_feature_train_tf_l.append(
                g_output_feature_train_tf)
            self.g_output_attribute_train_tf_l.append(
                g_output_attribute_train_tf)

        self.g_feature_train = torch.cat(
            self.g_output_feature_train_tf_l,
            dim=1)
        self.g_attribute_train = torch.cat(
            self.g_output_attribute_train_tf_l,
            dim=1)

        self.d_fake_train_tf = self.discriminator(
            self.g_feature_train,
            self.g_attribute_train)
        self.attr_d_fake_train_tf = self.attr_discriminator(
            self.g_attribute_train)

        # generate training route (real)
        self.real_feature_pl = torch.cat(
            self.data_feature,
            dim=1)
        self.real_attribute_pl = torch.cat(
            self.data_attribute,
            dim=1)
        self.d_real_train_tf = self.discriminator(
            self.real_feature_pl,
            self.real_attribute_pl)
        self.attr_d_real_train_tf = self.attr_discriminator(
            self.real_attribute_pl)

        return self.d_fake_train_tf, self.attr_d_fake_train_tf,\
            self.d_real_train_tf, self.attr_d_real_train_tf

    def sample_from(self,
                    real_attribute_input_noise,
                    addi_attribute_input_noise,
                    feature_input_noise,
                    feature_input_data,
                    gen_flag_dims,
                    batch_size=32):
        features = []
        attributes = []
        gen_flags = []
        lengths = []
        round_ = int(math.ceil(float(feature_input_noise.shape[0]) / batch_size))
        assert self.training is False, "please call .eval() on the model"
        self.generator.eval()
        for i in range(round_):
            (feature, attribute, gen_flag, length, _) = \
                self.generator(real_attribute_input_noise[i * batch_size:
                                                          (i + 1) * batch_size],
                               addi_attribute_input_noise[i * batch_size:
                                                          (i + 1) * batch_size],
                               feature_input_noise[i * batch_size:
                                                   (i + 1) * batch_size],
                               feature_input_data[i * batch_size:
                                                  (i + 1) * batch_size])
            features.append(feature)
            attributes.append(attribute)
            gen_flags.append(gen_flag)
            lengths.append(length)
        features = torch.cat(features, dim=0)
        attributes = torch.cat(attributes, dim=0)
        gen_flags = torch.cat(gen_flags, dim=0)
        lengths = torch.cat(lengths, dim=0)
        gen_flags = gen_flags[:, :, 0]

        features = features.detach().numpy()
        attributes = attributes.detach().numpy()
        gen_flags = gen_flags.detach().numpy()
        lengths = lengths.detach().numpy()

        features = np.delete(features, gen_flag_dims, axis=2)

        return features, attributes, gen_flags, lengths
