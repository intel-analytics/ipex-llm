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
import pickle
import numpy as np
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule

from .util import add_gen_flag, normalize_per_sample, renormalize_per_sample


class DoppelGANgerDataModule(LightningDataModule):
    '''
    Note that for now, we will still follow the Dataset format stated in
    https://github.com/fjxmlzn/DoppelGANger#dataset-format.

    Please notice that this module can not work alone without doppelganger_torch.
    '''
    def __init__(self,
                 sample_len,
                 real_data,
                 feature_outputs,
                 attribute_outputs,
                 batch_size=32):
        super().__init__()
        self.sample_len = sample_len
        self.batch_size = batch_size

        # load data from data_dir
        # ===================================================================================
        data_all = real_data['data_feature']
        data_attribute = real_data['data_attribute']
        data_gen_flag = real_data['data_gen_flag']
        data_feature_outputs = feature_outputs
        data_attribute_outputs = attribute_outputs
        self.num_real_attribute = len(data_attribute_outputs)
        self.num_feature_dim = len(data_feature_outputs)

        # normalize data (use this only if you want to use additional attribute(max, min))
        # actually, no additional attribute has not been fully tested now
        # ===================================================================================
        (data_feature, data_attribute, data_attribute_outputs,
            real_attribute_mask) = normalize_per_sample(
            data_all, data_attribute, data_feature_outputs,
            data_attribute_outputs)

        # add generation flag to features
        # ===================================================================================
        data_feature, data_feature_outputs = add_gen_flag(
            data_feature, data_gen_flag, data_feature_outputs, self.sample_len)

        # will be used in model init
        self.data_feature_outputs = data_feature_outputs
        self.data_attribute_outputs = data_attribute_outputs
        self.real_attribute_mask = real_attribute_mask

        # prepare input meta data
        # ===================================================================================
        total_generate_num_sample = data_feature.shape[0]
        from bigdl.nano.utils.common import invalidInputError
        if data_feature.shape[1] % self.sample_len != 0:
            invalidInputError(False, "length must be a multiple of sample_len")
        self.length = int(data_feature.shape[1] / self.sample_len)

        # will be used in dataset init
        self.data_feature = data_feature
        self.data_attribute = data_attribute

    def train_dataloader(self):
        self.data_feature = torch.from_numpy(self.data_feature).float()
        self.data_attribute = torch.from_numpy(self.data_attribute).float()
        dataset = CustomizedDataset(self.data_feature,
                                    self.data_attribute)
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=True)


class CustomizedDataset(Dataset):
    def __init__(self,
                 data_feature,
                 data_attribute):
        self.data_feature = data_feature
        self.data_attribute = data_attribute

    def __len__(self):
        return self.data_feature.shape[0]

    def __getitem__(self, index):
        return self.data_feature[index],\
            self.data_attribute[index]
