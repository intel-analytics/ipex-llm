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
#
# Copyright (c) 2018 CMU Locus Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This file is adapted from
# https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
# https://github.com/locuslab/TCN/blob/master/TCN/adding_problem/add_test.py

from tokenize import group
import numpy as np
import tensorflow as tf
from bigdl.nano.tf.keras import Model
from tensorflow.keras.layers import Conv1D, BatchNormalization,\
    Activation, Dropout, Input, Layer


class TemporalBlock(Layer):
    def __init__(self, dilation_rate, nb_filters, kernel_size=1, strides=1,
                 padding='casual', dropout_rate=0.0, repo_initialization=True):
        super(TemporalBlock, self).__init__()
        if repo_initialization:
            init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        else:
            init = tf.keras.initializers.HeUniform()

        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.repo_initialization = repo_initialization

        # block1
        self.conv1 = Conv1D(filters=nb_filters, kernel_size=kernel_size, strides=strides,
                            dilation_rate=dilation_rate, padding=padding,
                            kernel_initializer=init)
        self.batch1 = BatchNormalization(axis=-1)
        self.ac1 = Activation('relu')
        self.drop1 = Dropout(rate=dropout_rate)

        # block2
        self.conv2 = Conv1D(filters=nb_filters, kernel_size=kernel_size, strides=strides,
                            dilation_rate=dilation_rate, padding=padding,
                            kernel_initializer=init)
        self.batch2 = BatchNormalization(axis=-1)
        self.ac2 = Activation('relu')
        self.drop2 = Dropout(rate=dropout_rate)

        self.downsample = Conv1D(filters=nb_filters, kernel_size=1,
                                 padding='same', kernel_initializer=init)
        self.ac3 = Activation('relu')

    def call(self, x, training=False):
        prev_x = x
        out = self.conv1(x)
        out = self.batch1(out)
        out = self.ac1(out)
        out = self.drop1(out) if training else out

        out = self.conv2(out)
        out = self.batch2(out)
        out = self.ac2(out)
        out = self.drop2(out) if training else out

        if prev_x.shape[-1] != out.shape[-1]:    # match the dimention
            prev_x = self.downsample(prev_x)

        return self.ac3(prev_x + out)            # skip connection

    def get_config(self):
        return {"dilation_rate": self.dilation_rate,
                "nb_filters": self.nb_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dropout_rate": self.dropout_rate,
                "repo_initialization": self.repo_initialization}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TemporalConvNet(Model):
    def __init__(self,
                 past_seq_len,
                 future_seq_len,
                 input_feature_num,
                 output_feature_num,
                 num_channels=[30]*8,
                 kernel_size=3,
                 dropout=0.1,
                 repo_initialization=True):

        # num_channels is a list contains hidden sizes of Conv1D
        super(TemporalConvNet, self).__init__()

        self.past_seq_len = past_seq_len
        self.future_seq_len = future_seq_len
        self.input_feature_num = input_feature_num
        self.output_feature_num = output_feature_num
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.dropout = dropout
        self.repo_initialization = repo_initialization

        if repo_initialization:
            init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        else:
            init = tf.keras.initializers.HeUniform()

        # initialize model
        self.network = []

        # The model contains "num_levels" TemporalBlock
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_rate = 2 ** i
            self.network.append(TemporalBlock(dilation_rate, num_channels[i], kernel_size,
                                              padding='causal', dropout_rate=dropout))
        self.network.append(TemporalBlock(dilation_rate, self.output_feature_num, kernel_size,
                                          padding='causal', dropout_rate=dropout))

        self.linear = tf.keras.layers.Dense(future_seq_len, kernel_initializer=init)
        self.permute = tf.keras.layers.Permute((2, 1))

    def call(self, x, training=False):
        for layer in self.network:
            y = layer(x, training=training)
        y = self.permute(y)
        y = self.linear(y)
        y = self.permute(y)
        return y

    def get_config(self):
        return {"past_seq_len": self.past_seq_len,
                "future_seq_len": self.future_seq_len,
                "input_feature_num": self.input_feature_num,
                "output_feature_num": self.output_feature_num,
                "kernel_size": self.kernel_size,
                "num_channels": self.num_channels,
                "dropout": self.dropout,
                "repo_initialization": self.repo_initialization}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def model_creator(config):
    if config.get("num_channels"):
        num_channels = config["num_channels"]
    else:
        n_hid = config["nhid"] if config.get("nhid") else 30
        levels = config["levels"] if config.get("levels") else 8
        num_channels = [n_hid] * (levels - 1)

    model = TemporalConvNet(past_seq_len=config["past_seq_len"],
                            future_seq_len=config["future_seq_len"],
                            input_feature_num=config["input_feature_num"],
                            output_feature_num=config["output_feature_num"],
                            num_channels=num_channels.copy(),
                            kernel_size=config.get("kernel_size", 7),
                            dropout=config.get("dropout", 0.2),
                            repo_initialization=config.get("repo_initialization", True))
    inputs = Input(shape=(config["past_seq_len"], config["input_feature_num"]))
    # init weights matrix
    model(inputs)
    learning_rate = config.get('lr', 1e-3)
    model.compile(optimizer=getattr(tf.keras.optimizers,
                                    config.get("optim", "Adam"))(learning_rate),
                  loss=config.get("loss", "mse"),
                  metrics=[config.get("metics", "mse")])
    return model
