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
import sys
from bigdl.util.common import JavaValue
from bigdl.util.common import callBigDlFunc
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


class ChannelNormalizer(Preprocessing):
    """
     image norm
    """
    def __init__(self, meanR, meanG, meanB, stdR, stdG, stdB, bigdl_type="float"):
        super(ChannelNormalizer, self).__init__(bigdl_type, meanR, meanG, meanB, stdR, stdG, stdB)

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