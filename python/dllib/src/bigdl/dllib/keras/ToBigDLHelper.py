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

from math import ceil

import bigdl.dllib.nn.initialization_method as BInit
from bigdl.dllib.utils.log4Error import *
from bigdl.dllib.optim.optimizer import L1L2Regularizer as BRegularizer


def to_bigdl_2d_ordering(order):
    if order == "tf":
        return "NHWC"
    elif order == "th":
        return "NCHW"
    else:
        invalidInputError(False, "Unsupported dim_ordering: %s" % order)


def to_bigdl_3d_ordering(order):
    if order == "tf":
        return "channel_last"
    elif order == "th":
        return "channel_first"
    else:
        invalidInputError(False, "Unsupported dim_ordering: %s" % order)


def to_bigdl_3d_padding(border_mode):
    if border_mode == "valid":
        return 0, 0, 0
    elif border_mode == "same":
        return -1, -1, -1
    else:
        invalidInputError(False, "Unsupported border mode: %s" % border_mode)


def __calculate_2d_same_padding(x, kx, dx, dilation_x):
    return int(ceil((x * (dx - 1) + dilation_x * (kx - 1) - dx + 1) / 2))


def to_bigdl_2d_padding(border_mode, *args):
    if border_mode == "same":
        if len(args) == 0:  # if -1 for same padding is supported
            return -1, -1
        # calculate padding by given parameters
        elif len(args) == 4:  # used by 1d layers constructed from 2d, just need one pad
            h, kh, dh, dilation_h = args
            pad_h = __calculate_2d_same_padding(h, kh, dh, dilation_h)
            return pad_h, 0
        elif len(args) == 8:
            h, kh, dh, dilation_h, w, kw, dw, dilation_w = args
            pad_h = __calculate_2d_same_padding(h, kh, dh, dilation_h)
            pad_w = __calculate_2d_same_padding(w, kw, dw, dilation_w)
            return pad_h, pad_w
    elif border_mode == "valid":
        return 0, 0
    else:
        invalidInputError(False, "Unsupported border mode: %s" % border_mode)


def to_bigdl_init(kinit_method):  # kinit_method is a string
    init = None
    if kinit_method == "glorot_uniform":
        init = BInit.Xavier()
    elif kinit_method == "one":
        init = BInit.Ones()
    elif kinit_method == "zero":
        init = BInit.Zeros()
    elif kinit_method == "uniform":
        init = BInit.RandomUniform(lower=-0.05, upper=0.05)
    elif kinit_method == "normal":
        init = BInit.RandomNormal(mean=0.0, stdv=0.05)
    else:
        invalidInputError(False, "Unsupported init type: %s" % kinit_method)
    return init


def to_bigdl_reg(reg):  # reg is a dict
    if reg:
        return BRegularizer(reg['l1'], reg['l2'])
    else:
        return None
