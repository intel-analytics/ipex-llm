#
# Copyright 2018 Analytics Zoo Authors.
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

import sys

from ..engine.topology import ZooKerasLayer
from bigdl.util.common import callBigDlFunc, JTensor

if sys.version >= '3':
    long = int
    unicode = str


class BatchNormalization(ZooKerasLayer):
    """
    Batch normalization layer.
    Normalize the activations of the previous layer at each batch, i.e. applies a transformation
    that maintains the mean activation close to 0 and the activation standard deviation close to 1.
    It is a feature-wise normalization, each feature map in the input will be normalized separately.
    The input of this layer should be 4D or 2D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    epsilon: Small float > 0. Fuzz parameter. Default is 0.001.
    momentum: Float. Momentum in the computation of the exponential average of the mean and
              standard deviation of the data, for feature-wise normalization. Default is 0.99.
    beta_init: Name of the initialization function for shift parameter. Default is 'zero'.
    gamma_init: Name of the initialization function for scale parameter. Default is 'one'.
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last).
                  Default is 'th'. For 'th', axis along which to normalize is 1.
                  For 'tf', axis is 3.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> batchnormalization = BatchNormalization(input_shape=(3, 12, 12), name="bn1")
    creating: createZooKerasBatchNormalization
    """
    def __init__(self, epsilon=0.001, mode=0, axis=1, momentum=0.99, beta_init="zero",
                 gamma_init="one", dim_ordering="th", input_shape=None, **kwargs):
        if mode != 0:
            raise ValueError("For BatchNormalization, only mode=0 is supported for now")
        if dim_ordering == "th" and axis != 1:
            raise ValueError("For BatchNormalization with th dim ordering, "
                             "only axis=1 is supported for now")
        if dim_ordering == "tf" and axis != -1 and axis != 3:
            raise ValueError("For BatchNormalization with tf dim ordering, "
                             "only axis=-1 is supported for now")
        super(BatchNormalization, self).__init__(None,
                                                 float(epsilon),
                                                 float(momentum),
                                                 beta_init,
                                                 gamma_init,
                                                 dim_ordering,
                                                 list(input_shape) if input_shape else None,
                                                 **kwargs)

    def set_running_mean(self, running_mean):
        """
        Set the running mean of the BatchNormalization layer.
        :param running_mean: a Numpy array.
        """
        callBigDlFunc(self.bigdl_type, "setRunningMean",
                      self.value, JTensor.from_ndarray(running_mean))
        return self

    def set_running_std(self, running_std):
        """
        Set the running variance of the BatchNormalization layer.
        :param running_std: a Numpy array.
        """
        callBigDlFunc(self.bigdl_type, "setRunningStd",
                      self.value, JTensor.from_ndarray(running_std))
        return self

    def get_running_mean(self):
        """
        Get the running meaning of the BatchNormalization layer.
        """
        return callBigDlFunc(self.bigdl_type, "getRunningMean",
                             self.value).to_ndarray()

    def get_running_std(self):
        """
        Get the running variance of the BatchNormalization layer.
        """
        return callBigDlFunc(self.bigdl_type, "getRunningStd",
                             self.value).to_ndarray()
