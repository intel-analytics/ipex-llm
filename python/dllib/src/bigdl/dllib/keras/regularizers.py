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


from bigdl.dllib.optim.optimizer import *


def l1(l1=0.01):
    return L1Regularizer(l1=l1)


def l2(l2=0.01):
    return L2Regularizer(l2=l2)


def l1l2(l1=0.01, l2=0.01):
    return L1L2Regularizer(l1=l1, l2=l2)
