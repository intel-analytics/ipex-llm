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

import numpy as np
import tensorflow as tf

layers = tf.keras.layers

class TemporalBlock(tf.keras.Model):
    def __init__(self, dilation_rate, nb_filters, kernel_size=1, strides=1,
        padding='same', dropout_rate=0.0, repo_initialization=True):
        super(TemporalBlock, self).__init__()
        if repo_initialization:
            init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        else:
            init = tf.keras.initializers.HeUniform()

		# block1
        self.conv1 = layers.Conv1D(filters=nb_filters, kernel_size=kernel_size, strides = strides,
				                   dilation_rate=dilation_rate, padding=padding, kernel_initializer=init)
        self.batch1 = layers.BatchNormalization(axis=-1)
        self.ac1 = layers.Activation('relu') 
        self.drop1 = layers.Dropout(rate=dropout_rate)

		# block2
        self.conv2 = layers.Conv1D(filters=nb_filters, kernel_size=kernel_size, strides = strides,
						           dilation_rate=dilation_rate, padding=padding, kernel_initializer=init)
        self.batch2 = layers.BatchNormalization(axis=-1)		
        self.ac2 = layers.Activation('relu')
        self.drop2 = layers.Dropout(rate=dropout_rate)

        self.downsample = layers.Conv1D(filters=nb_filters, kernel_size=1, 
									    padding='same', kernel_initializer=init)
        self.ac3 = layers.Activation('relu')


    def call(self, x, training):
        prev_x = x
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.ac1(x)
        x = self.drop1(x) if training else x

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.ac2(x)
        x = self.drop2(x) if training else x

        if prev_x.shape[-1] != x.shape[-1]:    # match the dimention
            prev_x = self.downsample(prev_x)

        return self.ac3(prev_x + x)            # skip connection

class TemporalConvNet(tf.keras.Model):
    def __init__(self,
                 future_seq_len,
                 output_feature_num,
				 num_channels,
				 kernel_size=3,
				 dropout=0.1,
				 repo_initialization=True):
        # num_channels is a list contains hidden sizes of Conv1D
        super(TemporalConvNet, self).__init__()
        num_channels.append(output_feature_num)

        if repo_initialization:
            init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        else:
            init = tf.keras.initializers.HeUniform()

        # initialize model
        model = tf.keras.Sequential()

        # The model contains "num_levels" TemporalBlock
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_rate = 2 ** i
            model.add(TemporalBlock(dilation_rate, num_channels[i], kernel_size,
                      padding='causal', dropout_rate=dropout))

        self.network = model
        self.linear = tf.keras.layers.Dense(future_seq_len, kernel_initializer=init)
        self.permute = tf.keras.layers.Permute((2, 1))

    def call(self, x, training):
        y = self.network(x, training=training)
        y = self.permute(y)
        y = self.linear(y)
        y = self.permute(y)
        return y
