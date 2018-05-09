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

from bigdl.util.common import *
from zoo.feature.common import Preprocessing

if sys.version >= '3':
    long = int
    unicode = str


class Resize(Preprocessing):
    """
     image resize
    """
    def __init__(self, resizeH, resizeW, bigdl_type="float"):
        super(Resize, self).__init__(bigdl_type, resizeH, resizeW)


class ChannelNormalize(Preprocessing):
    """
    image channel normalize
    :param mean_r mean value in R channel
    :param mean_g mean value in G channel
    :param meanB_b mean value in B channel
    :param std_r std value in R channel
    :param std_g std value in G channel
    :param std_b std value in B channel
    """
    def __init__(self, mean_r, mean_b, mean_g, std_r=1.0, std_g=1.0, std_b=1.0, bigdl_type="float"):
        self.value = callBigDlFunc(
            bigdl_type, "createImgChannelNormalizer", mean_r, mean_g, mean_b, std_r, std_g, std_b)


class MatToTensor(Preprocessing):
    """
    MatToTensor
    """
    def __init__(self, bigdl_type="float"):
        super(MatToTensor, self).__init__(bigdl_type)


class CenterCrop(Preprocessing):
    """
    CenterCrop
    """
    def __init__(self, cropWidth, cropHeight, bigdl_type="float"):
        super(CenterCrop, self).__init__(bigdl_type, cropWidth, cropHeight)
