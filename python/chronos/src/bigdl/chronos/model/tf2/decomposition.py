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

import tensorflow as tf
from tensorflow.keras.layers import Layer
from bigdl.nano.tf.keras import Model


class moving_avg(Layer):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = tf.keras.layers.AveragePooling1D(pool_size=kernel_size, strides=stride,
                                                    padding='valid',
                                                    data_format='channels_first')
        self.permute = tf.keras.layers.Permute((2, 1))

    def call(self, x):
        # padding on the both ends of time series
        front = tf.tile(x[:, 0:1, :], [1, (self.kernel_size - 1) // 2, 1])
        end = tf.tile(x[:, -1:, :], [1, self.kernel_size // 2, 1])
        x = tf.concat([front, x, end], axis=1)
        x = self.avg(self.permute(x))
        x = self.permute(x)
        return x


class series_decomp(Layer):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def call(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DecompositionTSModel(Model):

    def __init__(self, models, kernel_size=25):
        """
        Build a Decomposition model wrapper.

        :param models: tuple, (model, model_copy) two basic forecaster models.
        :param kernel_size: int, Specify the kernel size in moving average.
        """
        super(DecompositionTSModel, self).__init__()
        self.decompsition = series_decomp(kernel_size)
        self.linear_seasonal = models[0]
        self.linear_trend = models[1]

    def call(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_output = self.linear_seasonal(seasonal_init)
        trend_output = self.linear_trend(trend_init)

        x = seasonal_output + trend_output
        return x
