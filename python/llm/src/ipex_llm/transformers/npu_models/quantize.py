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
from torch import Tensor


def c_round(x: Tensor):
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)

def update_scale_grid_search(x: Tensor, iscale: Tensor, min_max: list, N: int = 128 + 1):
    iscale = iscale.unsqueeze(1)

    assert N % 2 == 1, "Please check whether N: odd number"
    rng_dump = 0.05  # 0.05 / 1.
    z_val = 2e-4

    device = iscale.device
    dtype = iscale.dtype
    ###############################
    W_q = c_round(x * iscale).clamp(min_max[0], min_max[1])
    n_clusters = W_q.shape[0]
    rng = torch.abs(iscale).mean() * rng_dump if (rng_dump < 1.0) else rng_dump

    iscale_shifted = (
        torch.linspace(-rng, rng, N)[None, :]
        .to(dtype=dtype, device=device)
        .repeat(n_clusters, 1)
    ) + iscale

    # Safe inverse
    iscale_shifted[
        torch.logical_and(iscale_shifted >= 0, torch.abs(iscale_shifted) <= z_val)
    ] = z_val
    iscale_shifted[
        torch.logical_and(iscale_shifted < 0, torch.abs(iscale_shifted) <= z_val)
    ] = -z_val

    err = torch.empty([n_clusters, N], dtype=dtype, device=device)
    for i in range(N):
        W_r = W_q * iscale_shifted[:, i][:, None]
        err[:, i] = torch.abs(x - W_r).mean(axis=1, keepdim=True).squeeze()

    ind_r = torch.argmin(err, axis=1).to(torch.int32)
    ind_c = torch.arange(len(ind_r), dtype=torch.int32, device=device)
    iscale_b = iscale_shifted[ind_c, ind_r]
    scale_b = 1.0 / iscale_b
    iscale_b = iscale_b.unsqueeze(1)

    # test with original
    # scale_b = (1.0 / iscale).squeeze()
    # qweights = (c_round(x * iscale)).clamp(-8.0, 7.0).to(torch.int8) # m * n

    # obtain qwights based on scale_b
    qweights = (c_round(x * iscale_b)).clamp(min_max[0], min_max[1]).to(torch.int8) # m * n
    qweights = qweights.reshape(x.shape[0], -1 , 2) # m * n/2 * 2
    low_bit, high_bit = qweights.split(1, dim=-1)
    high_bit = high_bit.squeeze().view(torch.int8)
    low_bit = low_bit.squeeze().view(torch.int8)
    high_bit = high_bit << 4
    low_bit = low_bit & 0x0f
    qweights = high_bit | low_bit

    return qweights.view(torch.uint8), scale_b.to(torch.float16)
