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
# Some parts of this file is adapted from
# https://github.com/pytorch/pytorch/blob/v2.1.0/torch/cuda/amp/autocast_mode.py
#
# From PyTorch:

# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert,
#                                                   Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# From Caffe2:

# Copyright (c) 2016-present, Facebook Inc. All rights reserved.

# All contributions by Facebook:
# Copyright (c) 2016 Facebook Inc.

# All contributions by Google:
# Copyright (c) 2015 Google Inc.
# All rights reserved.

# All contributions by Yangqing Jia:
# Copyright (c) 2015 Yangqing Jia
# All rights reserved.

# All contributions by Kakao Brain:
# Copyright 2019-2020 Kakao Brain

# All contributions by Cruise LLC:
# Copyright (c) 2022 Cruise LLC.
# All rights reserved.

# All contributions from Caffe:
# Copyright(c) 2013, 2014, 2015, the respective contributors
# All rights reserved.

# All other contributions:
# Copyright(c) 2015, 2016 the respective contributors
# All rights reserved.

# Caffe2 uses a copyright model similar to Caffe: each contributor holds
# copyright over their contributions to Caffe2. The project versioning records
# all such contribution and copyright details. If a contributor wants to further
# mark their specific copyright on a particular contribution, they should
# indicate their copyright solely in the commit message of the change when it is
# committed.

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
#    and IDIAP Research Institute nor the names of its contributors may be
#    used to endorse or promote products derived from this software without
#    specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
import collections
import functools

import torch

try:
    import numpy as np

    HAS_NUMPY = True
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]
from typing import Any


def _cast(value, dtype):
    if isinstance(value, torch.Tensor):
        is_eligible = (
            value.is_floating_point()
            and value.is_xpu
            and (value.dtype is not torch.float64)
        )
        return value.to(dtype) if is_eligible else value
    elif isinstance(value, (str, bytes)):
        return value
    elif HAS_NUMPY and isinstance(value, np.ndarray):
        return value
    elif isinstance(value, collections.abc.Mapping):
        return {_cast(k, dtype): _cast(v, dtype) for k, v in value.items()}
    elif isinstance(value, collections.abc.Iterable):
        iterable = (_cast(v, dtype) for v in value)
        if isinstance(value, (list, tuple)):
            return type(value)(iterable)
        else:
            return iterable
    else:
        return value


def custom_fwd(fwd=None, *, cast_inputs=None):
    """
    Helper decorator for ``forward`` methods of custom autograd functions (subclasses of
    :class:`torch.autograd.Function`).  See the :ref:`example page<amp-custom-examples>`
    for more detail.

    Args:
        cast_inputs (:class:`torch.dtype` or None, optional, default=None):  If not ``None``,
            when ``forward`` runs in an autocast-enabled region, casts incoming
            floating-point CUDA Tensors to the target dtype (non-floating-point Tensors
            are not affected),
            then executes ``forward`` with autocast disabled.
            If ``None``, ``forward``'s internal ops execute with the current autocast state.

    .. note::
        If the decorated ``forward`` is called outside an autocast-enabled region,
        :func:`custom_fwd<custom_fwd>` is a no-op and ``cast_inputs`` has no effect.
    """
    if fwd is None:
        return functools.partial(custom_fwd, cast_inputs=cast_inputs)

    @functools.wraps(fwd)
    def decorate_fwd(*args, **kwargs):
        args[0]._dtype = torch.xpu.get_autocast_xpu_dtype()
        if cast_inputs is None:
            args[0]._fwd_used_autocast = torch.xpu.is_autocast_xpu_enabled()
            return fwd(*args, **kwargs)
        else:
            autocast_context = torch.xpu.is_autocast_xpu_enabled()
            args[0]._fwd_used_autocast = False
            if autocast_context:
                with torch.xpu.autocast(enabled=False):
                    return fwd(*_cast(args, cast_inputs), **_cast(kwargs, cast_inputs))
            else:
                return fwd(*args, **kwargs)

    return decorate_fwd


# Autograd ensures incoming gradients are the same type as forward outputs.  Allowing a separate
# cast_inputs argument on custom_bwd is unnecessary and could cause errors if it doesn't match
# cast_inputs supplied to custom_fwd.
def custom_bwd(bwd):
    """
    Helper decorator for backward methods of custom autograd functions (subclasses of
    :class:`torch.autograd.Function`).
    Ensures that ``backward`` executes with the same autocast state as ``forward``.
    See the :ref:`example page<amp-custom-examples>` for more detail.
    """

    @functools.wraps(bwd)
    def decorate_bwd(*args, **kwargs):
        with torch.xpu.autocast(enabled=args[0]._fwd_used_autocast, dtype=args[0]._dtype):
            return bwd(*args, **kwargs)

    return decorate_bwd
