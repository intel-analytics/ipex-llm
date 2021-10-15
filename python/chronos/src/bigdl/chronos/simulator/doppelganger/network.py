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

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from enum import Enum
from .op import linear, flatten
from .output import OutputType, Normalization


class Discriminator(nn.Module):
    '''
    pytorch nn.Module implementation for gan.network.Discriminator

    `input_size` is an extra param that need to be set during the init
    due to the different stype between tf and torch
    `input_size` should be the sum of input_feature and input_attribute
    flattened length
    '''
    def __init__(self, input_size,
                 num_layers=5, num_units=200):
        super().__init__()

        self.num_layers = num_layers
        self.num_units = num_units

        layers_dim = [input_size] + [num_units]*(num_layers-1) + [1]
        mlp_layers = []
        for i in range(num_layers):
            mlp_layers.append(linear(input_size=layers_dim[i],
                                     output_size=layers_dim[i+1]))
            if i < num_layers - 1:
                mlp_layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, input_feature, input_attribute):
        # input_feature.shape = (batch_size, max_time_stamp, feature_dim) ?
        # input_attribute.shape = (batch_size, all_attribute_out_dim) (flattened)
        # output feature: (batch_size, 1)
        input_feature = flatten(input_feature)
        input_attribute = flatten(input_attribute)
        x = torch.cat([input_feature, input_attribute], dim=1)
        x = self.mlp(x)
        x = torch.squeeze(x, 1)
        return x


class AttrDiscriminator(nn.Module):
    '''
    pytorch nn.Module implementation for gan.network.AttrDiscriminator

    `input_size` is an extra param that need to be set during the init
    due to the different stype between tf and torch
    `input_size` should be the input_attribute flattened length
    '''
    def __init__(self, input_size,
                 num_layers=5, num_units=200):
        super().__init__()

        self.num_layers = num_layers
        self.num_units = num_units

        layers_dim = [input_size] + [num_units]*(num_layers-1) + [1]
        mlp_layers = []
        for i in range(num_layers):
            mlp_layers.append(linear(input_size=layers_dim[i],
                                     output_size=layers_dim[i+1]))
            if i < num_layers - 1:
                mlp_layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, input_attribute):
        # input_attribute.shape = (batch_size, all_attribute_out_dim) (flattened)
        # output feature: (batch_size, 1)
        input_attribute = flatten(input_attribute)
        x = self.mlp(input_attribute)
        x = torch.squeeze(x, 1)
        return x


class RNNInitialStateType(Enum):
    '''
    same as gan.network.RNNInitialStateType
    '''
    ZERO = "ZERO"
    RANDOM = "RANDOM"
    VARIABLE = "VARIABLE"


class DoppelGANgerGenerator(nn.Module):
    '''
    pytorch nn.Module implementation for gan.network.DoppelGANgerGenerator
    '''
    def __init__(self, feed_back, noise,
                 feature_outputs, attribute_outputs, real_attribute_mask,
                 sample_len,
                 attribute_num_units=100, attribute_num_layers=3,
                 feature_num_units=100, feature_num_layers=2,
                 attribute_input_noise_dim=5, addi_attribute_input_noise_dim=5,
                 feature_input_noise_dim=5,
                 attribute_dim=None,
                 initial_state=RNNInitialStateType.RANDOM,
                 initial_stddev=0.02):
        '''
        Some explanation of params:
        :param feedback: bool, this is a typical RNN mode
        :param noise: bool, this is True for most of the case
               since it is the version of Figure 7.
        :param attribute_input_noise_dim: defaults to 5.
        :param addi_attribute_input_noise_dim: defaults to 5.
        :param attribute_dim: only used for Min/Max Generator(MLP)
        :param feature_outputs: A list of Output objects, indicating the
               dimension, type, normalization of each feature
        :param attribute_outputs A list of Output objects, indicating the
               dimension, type, normalization of each attribute
        :param sample_len: parameter S in Section 4.1 in the paper
               refer to https://github.com/fjxmlzn/DoppelGANger/issues/9
        :param real_attribute_mask: List of True/False, the length equals the
               number of attributes. False if the attribute is (max-min)/2 or
               (max+min)/2, True otherwise
        '''
        super().__init__()

        # feedback or noise(Recommeneded)
        self.feed_back = feed_back
        self.noise = noise

        # parameter S in Section 4.1
        self.sample_len = sample_len

        # MLP generator layer num and hidden units
        self.attribute_num_units = attribute_num_units
        self.attribute_num_layers = attribute_num_layers
        # LSTM generator layer num and hidden units
        self.feature_num_units = feature_num_units
        self.feature_num_layers = feature_num_layers

        # abstract output meta information
        self.feature_outputs = feature_outputs
        self.attribute_outputs = attribute_outputs
        self.feature_out_dim = (np.sum([t.dim for t in feature_outputs]) *
                                self.sample_len)
        self.attribute_out_dim = np.sum([t.dim for t in attribute_outputs])

        # real attribute(all attribute other than min/max related)
        self.real_attribute_mask = real_attribute_mask

        # RNN init state
        self.initial_state = initial_state

        # same as line 137-165 in gan.network.DoppelGANgerGenerator
        # split the output meta information to real and additional(max/min)
        self.real_attribute_outputs = []
        self.addi_attribute_outputs = []
        self.real_attribute_out_dim = 0
        self.addi_attribute_out_dim = 0
        for i in range(len(self.real_attribute_mask)):
            if self.real_attribute_mask[i]:
                self.real_attribute_outputs.append(
                    self.attribute_outputs[i])
                self.real_attribute_out_dim += self.attribute_outputs[i].dim
            else:
                self.addi_attribute_outputs.append(
                    self.attribute_outputs[i])
                self.addi_attribute_out_dim += \
                    self.attribute_outputs[i].dim

        # real attribute should come first
        for i in range(len(self.real_attribute_mask) - 1):
            if (self.real_attribute_mask[i] is False and
                    self.real_attribute_mask[i + 1] is True):
                raise Exception("Real attribute should come first")

        # find the output id of gen flag in feature outputs
        self.gen_flag_id = None
        for i in range(len(self.feature_outputs)):
            if self.feature_outputs[i].is_gen_flag:
                self.gen_flag_id = i
                break
        if self.gen_flag_id is None:
            raise Exception("cannot find gen_flag_id")
        # gen flag can only be [0,1] or [1,0]
        # gen flag is explained in appendix B, right below table 8.
        if self.feature_outputs[self.gen_flag_id].dim != 2:
            raise Exception("gen flag output's dim should be 2")

        # prepare noise meta data (need more discussion or understanding)
        if attribute_dim is None:
            # attribute_dim is None for the 2 Generator(MLP), both take
            # noise(and nothing else) as input.
            all_attribute_dim = []
            all_discrete_attribute_dim = []
            if len(self.addi_attribute_outputs) > 0:
                self.all_attribute_input_noise_dim = [attribute_input_noise_dim,
                                                      addi_attribute_input_noise_dim]
                self.all_attribute_outputs = [self.real_attribute_outputs,
                                              self.addi_attribute_outputs]
                self.all_attribute_out_dim = [self.real_attribute_out_dim,
                                              self.addi_attribute_out_dim]
            else:
                self.all_attribute_input_noise_dim = [attribute_input_noise_dim]
                self.all_attribute_outputs = [self.real_attribute_outputs]
                self.all_attribute_out_dim = [self.real_attribute_out_dim]
        else:
            # attribute is not None for only one Generator(MLP)
            # attribute_dim is m for (A1, ..., Am)
            all_attribute_dim = [attribute_dim]
            all_discrete_attribute_dim = [attribute_dim]
            if len(self.addi_attribute_outputs) > 0:
                self.all_attribute_input_noise_dim = [addi_attribute_input_noise_dim]
                self.all_attribute_outputs = [self.addi_attribute_outputs]
                self.all_attribute_out_dim = [self.addi_attribute_out_dim]
            else:
                self.all_attribute_input_noise_dim = []
                self.all_attribute_outputs = []
                self.all_attribute_out_dim = []

        # prepare MLP geneartors(Metadata Generator and Min/Max Generator)
        self.mlp_1 = None
        # Metadata Generator if mlp_2 is not None, Min/Max Generator if mlp_2 is None
        self.mlp_2 = None
        # Min/Max Generator if not None
        for part_i in range(len(self.all_attribute_input_noise_dim)):
            # the loop will run at most 2 rounds
            # generate mlp each layer dim
            if len(all_discrete_attribute_dim) > 0:
                layers_dim = [self.all_attribute_input_noise_dim[part_i] +
                              all_discrete_attribute_dim[-1]]
            else:
                layers_dim = [self.all_attribute_input_noise_dim[part_i]]
            layers_dim += [self.attribute_num_units]*(self.attribute_num_layers-1)

            mlp_list = []
            for i in range(len(layers_dim) - 1):
                mlp_list.append(linear(layers_dim[i], layers_dim[i+1]))
                mlp_list.append(torch.nn.ReLU())
                mlp_list.append(nn.BatchNorm1d(layers_dim[i+1], eps=1e-05, momentum=0.1))
            mlp_list.append(linear(layers_dim[-1], self.all_attribute_out_dim[part_i]))

            assert part_i <= 1, 'part_i should smaller than 2!'
            if part_i == 0:
                self.mlp_1 = nn.Sequential(*mlp_list)
            if part_i == 1:
                self.mlp_2 = nn.Sequential(*mlp_list)
            all_discrete_attribute_dim.append(self.all_attribute_out_dim[part_i])

        # prepare the RNN for meaturements(features) generation
        self.rnn_network = nn.LSTM(input_size=self.attribute_out_dim + feature_input_noise_dim,
                                   hidden_size=self.feature_num_units,
                                   num_layers=self.feature_num_layers,
                                   batch_first=True)  # use batch_first to reduce many transpose

        # mlp_rnn_list = []
        # for i in range(self.sample_len):
        #     mlp_rnn_list.append(linear(self.feature_num_units, self.feature_out_dim))
        # self.mlp_rnn_list = nn.ModuleList(mlp_rnn_list)
        self.mlp_rnn = linear(self.feature_num_units, self.feature_out_dim)

    def _post_process_generated_attribute(self, sub_attribute_output, sub_all_attribute_outputs):
        # real attribute post-process
        current_idx = 0
        part_attribute = []
        part_discrete_attribute = []
        for i in range(len(sub_all_attribute_outputs)):
            output = sub_all_attribute_outputs[i]
            if output.type_ == OutputType.DISCRETE:
                sub_output = F.softmax(sub_attribute_output[:, current_idx:
                                                            current_idx+output.dim],
                                       dim=1)  # batch_size, output.dim
                sub_output_discrete = F.one_hot(torch.argmax(sub_output, dim=1),
                                                num_classes=output.dim)
                # batch_size, output.dim, 0/1
            elif output.type_ == OutputType.CONTINUOUS:
                if output.normalization == Normalization.ZERO_ONE:
                    sub_output = torch.sigmoid(sub_attribute_output[:, current_idx:
                                               current_idx+output.dim])
                elif (output.normalization == Normalization.MINUSONE_ONE):
                    sub_output = torch.tanh(sub_attribute_output[:, current_idx:
                                            current_idx+output.dim])
                else:
                    raise Exception("unknown normalization type")
                sub_output_discrete = sub_output
            else:
                raise Exception("unknown output type")
            part_attribute.append(sub_output)
            part_discrete_attribute.append(sub_output_discrete)
            current_idx += output.dim
        assert current_idx == np.sum([t.dim for t in sub_all_attribute_outputs])
        part_attribute = torch.cat(part_attribute, dim=1)
        part_discrete_attribute = torch.cat(part_discrete_attribute, dim=1)
        part_discrete_attribute = part_discrete_attribute.detach()
        # discrete attribute will be used as other generator's input
        assert part_attribute.ndim == 2
        assert part_discrete_attribute.ndim == 2
        return part_attribute, part_discrete_attribute

    def forward(self, attribute_input_noise, addi_attribute_input_noise,
                feature_input_noise, feature_input_data, attribute=None):
        # attribute_input_noise.shape = (batch_size, 5)
        # addi_attribute_input_noise.shape = (batch_size, 5)
        # feature_input_noise.shape = (batch_size, time, *) time is sample_time
        # feature_input_data.shape = (batch_size, time, *) or 2 dim
        # attribute.shape = (batch_size, real_attribute_out_dim)

        all_attribute_input_noise_dim = self.all_attribute_input_noise_dim
        all_attribute_outputs = self.all_attribute_outputs
        all_attribute_out_dim = self.all_attribute_out_dim

        all_attribute = []
        all_discrete_attribute = []
        if attribute is not None:
            all_attribute.append(attribute)
            all_discrete_attribute.append(attribute)

        attribute_output = None
        additional_attribute_output = None
        part_real_attribute = None  # post-processed attribute_output
        part_real_discrete_attribute = None  # post-processed attribute_output(discrete)
        part_additional_attribute = None  # post-processed additional_attribute_output
        part_additional_discrete_attribute = None
        # post-processed additional_attribute_output(discrete)
        # real_attribute_out_dim = sum of all real attribute dim
        # addi_attribute_out_dim = sum of all additional attribute dim
        if self.mlp_1 is not None and self.mlp_2 is not None:
            # assert there are real attr and fake attr
            assert len(all_attribute_outputs) == 2

            # generate real attr
            attribute_output = self.mlp_1(attribute_input_noise)
            # (batch_size, real_attribute_out_dim)
            part_real_attribute, part_real_discrete_attribute = \
                self._post_process_generated_attribute(attribute_output, all_attribute_outputs[0])

            # generate addi attr
            addi_attribute_generator_input = torch.cat([part_real_discrete_attribute,
                                                        addi_attribute_input_noise], dim=1)
            additional_attribute_output = self.mlp_2(addi_attribute_generator_input)
            # (batch_size, addi_attribute_out_dim)
            part_additional_attribute, part_additional_discrete_attribute = \
                self._post_process_generated_attribute(additional_attribute_output,
                                                       all_attribute_outputs[1])

        elif self.mlp_1 is not None and self.mlp_2 is None:
            additional_attribute_output = self.mlp_1(attribute_input_noise)
            # (batch_size, addi_attribute_out_dim)
            assert len(all_attribute_outputs) == 1
            part_additional_attribute, part_additional_discrete_attribute = \
                self._post_process_generated_attribute(additional_attribute_output,
                                                       all_attribute_outputs[0])

        if part_real_attribute is not None:
            all_attribute.append(part_real_attribute)
        if part_additional_attribute is not None:
            all_attribute.append(part_additional_attribute)
        if part_real_discrete_attribute is not None:
            all_discrete_attribute.append(part_real_discrete_attribute)
        if part_additional_discrete_attribute is not None:
            all_discrete_attribute.append(part_additional_discrete_attribute)

        all_attribute = torch.cat(all_attribute, dim=1)
        # will be used in Auxiliary Discriminator
        all_discrete_attribute = torch.cat(all_discrete_attribute, dim=1)
        # will be send to RNN, should with out grad
        all_discrete_attribute = all_discrete_attribute.detach()
        # just to make sure it is detached, maybe not needed

        # initial_state preparation
        if self.initial_state == RNNInitialStateType.ZERO:
            initial_state_c = torch.zeros(self.feature_num_layers,  # L (layer)
                                          feature_input_data.shape[0],  # N (batch_size)
                                          self.feature_num_units)  # H (hidden dim)
            initial_state_h = torch.zeros(self.feature_num_layers,  # L (layer)
                                          feature_input_data.shape[0],  # N (batch_size)
                                          self.feature_num_units)  # H (hidden dim)
        elif self.initial_state == RNNInitialStateType.RANDOM:
            initial_state_c = torch.randn(self.feature_num_layers,  # L (layer)
                                          feature_input_data.shape[0],  # N (batch_size)
                                          self.feature_num_units)  # H (hidden dim)
            initial_state_h = torch.randn(self.feature_num_layers,  # L (layer)
                                          feature_input_data.shape[0],  # N (batch_size)
                                          self.feature_num_units)  # H (hidden dim)
        elif self.initial_state == RNNInitialStateType.VARIABLE:
            raise NotImplementedError("RNNInitialStateType.VARIABLE has not been implemented!")
        else:
            raise NotImplementedError

        state = (initial_state_h, initial_state_c)
        batch_size = feature_input_data.shape[0]
        # time = feature_input_data.shape[1]//self.sample_len # max time length
        time = feature_input_noise.shape[1]   # sample_time
        # all_gen_flag.shape = (feature_out_dim, (batch_size, 1)) 1 means to geneate
        all_gen_flag = []
        # all_cur_argmax.shape = (feature_out_dim, (batch_size, 1)) 0 means to generate
        all_cur_argmax = []
        # gen_flag.shape = (batch_size, 1) 1 means to generate
        gen_flag = torch.ones(batch_size, 1)

        time_used = 0
        all_output = []
        for i in range(time):
            time_used = i + 1
            input_all = [all_discrete_attribute]  # (batch_size, attribute_out_dim)
            if self.noise:
                input_all.append(feature_input_noise[:, i])  # (batch_size, feature_out_dim)
            if self.feed_back:
                if feature_input_data.ndim == 3:
                    input_all.append(feature_input_data[:, i])  # (batch_size, feature_out_dim)
                else:
                    input_all.append(new_output)
            input_all = torch.cat(input_all, dim=1)
            # (batch_size, attribute_out_dim + feature_noise_dim)
            input_all = torch.unsqueeze(input_all, dim=1)
            # (batch_size, 1, attribute_out_dim + feature_noise_dim)
            out_rnn, (h_n, c_n) = self.rnn_network(input_all, state)
            state = (h_n, c_n)

            # out.shape = (batch_size, length=1, out_dim)
            # output_feature = self.mlp_rnn(out_rnn)
            # output_feature.shape = (batch_size, feature_out_dim)
            new_output_all = []
            current_idx = 0
            output_feature = self.mlp_rnn(out_rnn)
            for j in range(self.sample_len):
                for k in range(len(self.feature_outputs)):
                    output = self.feature_outputs[k]
                    sub_output = output_feature[:, current_idx:current_idx+output.dim]
                    if (output.type_ == OutputType.DISCRETE):
                        sub_output = F.softmax(sub_output, dim=1)
                    elif (output.type_ == OutputType.CONTINUOUS):
                        if (output.normalization ==
                                Normalization.ZERO_ONE):
                            sub_output = torch.sigmoid(sub_output)
                        elif (output.normalization ==
                                Normalization.MINUSONE_ONE):
                            sub_output = torch.tanh(sub_output)
                        else:
                            raise Exception("unknown normalization"
                                            " type")
                    else:
                        raise Exception("unknown output type")
                    new_output_all.append(sub_output)
                    current_idx += output.dim
            new_output = torch.cat(new_output_all, dim=1)
            all_output.append(new_output)

            for j in range(self.sample_len):
                cur_gen_flag = None  # (batch_size, 1) 1 means to generate
                all_gen_flag.append(gen_flag)
                cur_gen_flag = (torch.argmax(new_output_all[(j * len(self.feature_outputs) +
                                             self.gen_flag_id)], dim=1) == 0).float().view(-1, 1)
                all_cur_argmax.append(torch.argmax(new_output_all[(j * len(self.feature_outputs) +
                                                   self.gen_flag_id)], dim=1))
                gen_flag = gen_flag * cur_gen_flag
            if torch.max(gen_flag) == 0:
                break

        for i in range(time_used, time):
            all_output.append(torch.zeros(batch_size, self.feature_out_dim))
            for j in range(self.sample_len):
                all_gen_flag.append(torch.zeros(batch_size, 1))
                all_cur_argmax.append(torch.zeros(batch_size))
        feature = torch.stack(all_output, axis=1)
        # batch_size * time * (dim * sample_len)
        gen_flag = torch.stack(all_gen_flag, axis=1)
        # batch_size * (time * sample_len) * 1
        cur_argmax = torch.stack(all_cur_argmax, axis=1)
        # batch_size * (time * sample_len)
        length = torch.sum(gen_flag, dim=(1, 2))
        # batch_size
        gen_flag_t = gen_flag.view(batch_size, time, self.sample_len)
        # batch_size * time * sample_len
        gen_flag_t = torch.sum(gen_flag_t, dim=2)
        # batch_size * time * sample_len
        gen_flag_t = (gen_flag_t > 0.5).float()
        gen_flag_t = torch.unsqueeze(gen_flag_t, dim=2)
        # torch >= 1.8
        # gen_flag_t = torch.tile(gen_flag_t, [1, 1, self.feature_out_dim])
        # print("gen_flag_t old shape2:", gen_flag_t.shape)
        # torch < 1.8
        gen_flag_t = gen_flag_t.repeat(1, 1, self.feature_out_dim)
        # batch_size * time * (dim * sample_len)
        feature = feature * gen_flag_t
        feature = feature.view(batch_size, time * self.sample_len,
                               int(self.feature_out_dim / self.sample_len))
        # batch_size * (time * sample_len) * dim

        return feature, all_attribute, gen_flag, length, cur_argmax
