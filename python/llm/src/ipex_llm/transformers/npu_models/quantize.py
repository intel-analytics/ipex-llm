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
# https://github.com/mobiusml/hqq/blob/master/hqq/core/optimize.py
# which is licensed under Apache License 2.0:
#
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

import torch
import numpy as np
from torch import float32, float16, Tensor
from functools import partial
from typing import Union


def update_scale_grid_search(x: Tensor, scale: Tensor, min_max: list, N: int = 128 + 1):
    print(x.shape)
    print(scale.shape)

    assert N % 2 == 1, "Please check whether N: odd number"
    rng_dump = 0.05  # 0.05 / 1.
    z_val = 2e-4

    device = scale.device
    dtype = scale.dtype
    ###############################
    print("init scale shape is : ", scale.shape)
    W_q = (x / scale).clamp(min_max[0], min_max[1])
    n_clusters = W_q.shape[0]
    rng = torch.abs(scale).mean() * rng_dump if (rng_dump < 1.0) else rng_dump
    print("rng is : ", rng)

    scale_shifted = (
        torch.linspace(-rng, rng, N)[None, :]
        .to(dtype=dtype, device=device)
        .repeat(n_clusters, 1)
    )

    scale_shifted += scale

    # Safe inverse
    scale_shifted[
        torch.logical_and(scale_shifted >= 0, torch.abs(scale_shifted) <= z_val)
    ] = z_val
    scale_shifted[
        torch.logical_and(scale_shifted < 0, torch.abs(scale_shifted) <= z_val)
    ] = -z_val

    err = torch.empty([n_clusters, N], dtype=dtype, device=device)
    for i in range(N):
        W_r = W_q  * scale_shifted[:, i][:, None]
        err[:, i] = torch.abs(x - W_r).mean(axis=1, keepdim=True).squeeze()
        print(f"err [{i}] shape is ", err[i].shape)
    
    ind_r = torch.argmin(err, axis=1).to(torch.int32)
    ind_c = torch.arange(len(ind_r), dtype=torch.int32, device=device)
    scale_b = scale_shifted[ind_c, ind_r]

    # obtain qwights based on scale_b

    return scale_b, qweights
