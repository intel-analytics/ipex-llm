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
import numpy as np


def c_round(x: Tensor):
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


def update_scale_grid_search(x: Tensor, iscale: Tensor, min_max: list, N: int = 128 + 1):
    iscale = iscale.unsqueeze(1)

    assert N % 2 == 1, "Please check whether N: odd number"
    rng_dump = 0.05  # 0.05 / 1.

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


# Shrinking operator
def shrink_lp_op(x: Tensor, beta: float, lp_norm: float) -> Tensor:
    if lp_norm == 1:
        return torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - 1.0 / beta)
    else:
        return torch.sign(x) * torch.nn.functional.relu(
            torch.abs(x) - (1.0 / beta) * torch.pow(torch.abs(x), lp_norm - 1)
        )


def update_scale_hqq(x: Tensor, iscale: Tensor, min_max: list):
    iscale = iscale.unsqueeze(1)
    opt_params: dict = {"lp_norm": 0.7, "beta": 1e1, "kappa": 1.01, "iters": 20}
    lp_norm, beta, kappa, iters = (
        opt_params["lp_norm"],
        opt_params["beta"],
        opt_params["kappa"],
        opt_params["iters"],
    )
    z_val = 1e-4
    delta = 1e-4

    best_error = 1e4
    for i in range(iters):
        W_q = c_round(x * iscale).clamp(min_max[0], min_max[1])
        W_q_mask = W_q == 0
        W_q[W_q_mask] = delta
        W_r = W_q / iscale
        W_e = shrink_lp_op(x - W_r, beta, lp_norm)
        W_ = (x - W_e).clone()
        W_mask = torch.abs(W_) < z_val
        W_[W_mask] = z_val
        iscale, _ = torch.median(W_q / W_, axis=1, keepdim=True)
        beta *= kappa

        current_error = float(torch.abs(x - W_r).mean())
        print(i, current_error)
        print(iscale, torch.isinf(iscale).any(), torch.isnan(iscale).any())
        if current_error < best_error:
            best_error = current_error
        else:
            break
    
    scale_b = 1.0 / iscale
    qweights = (c_round(x * iscale)).clamp(min_max[0], min_max[1]).to(torch.int8) # m * n
    qweights = qweights.reshape(x.shape[0], -1 , 2) # m * n/2 * 2
    low_bit, high_bit = qweights.split(1, dim=-1)
    high_bit = high_bit.squeeze().view(torch.int8)
    low_bit = low_bit.squeeze().view(torch.int8)
    high_bit = high_bit << 4
    low_bit = low_bit & 0x0f
    qweights = high_bit | low_bit

    return qweights.view(torch.uint8), scale_b.to(torch.float16)


def update_scale_hqq_v2(x: Tensor, scale: Tensor, min_max: list):
    scale = scale.unsqueeze(1)
    opt_params: dict = {"lp_norm": 0.7, "beta": 1e1, "kappa": 1.01, "iters": 20}
    lp_norm, beta, kappa, iters = (
        opt_params["lp_norm"],
        opt_params["beta"],
        opt_params["kappa"],
        opt_params["iters"],
    )

    best_error = 1e4
    for i in range(iters):
        W_q = c_round(x / scale).clamp(min_max[0], min_max[1])
        W_q_mask = W_q != 0  # m, n
        sum_row = torch.sum(W_q_mask.int(), axis=1, keepdim=True) # m, 1
        W_r = W_q * scale
        W_e = shrink_lp_op(x - W_r, beta, lp_norm)
        W_ = (x - W_e).clone()
        tmp = W_ / W_q
        tmp[W_q == 0] = 0
        tmp = torch.sum(tmp, axis=1, keepdim=True) # m, 1
        scale = tmp / sum_row # m, 1
        beta *= kappa

        current_error = float(torch.abs(x - W_r).mean())
        print(i, current_error)
        if current_error < best_error:
            best_error = current_error
        else:
            break
    
    scale_b = scale
    qweights = (c_round(x / scale)).clamp(min_max[0], min_max[1]).to(torch.int8) # m * n
    qweights = qweights.reshape(x.shape[0], -1 , 2) # m * n/2 * 2
    low_bit, high_bit = qweights.split(1, dim=-1)
    high_bit = high_bit.squeeze().view(torch.int8)
    low_bit = low_bit.squeeze().view(torch.int8)
    high_bit = high_bit << 4
    low_bit = low_bit & 0x0f
    qweights = high_bit | low_bit

    return qweights.view(torch.uint8), scale_b.to(torch.float16)


# re-estimate the scale based on the inverse median: Only tested with axis==0
def update_scale_inverse_median(
    W_f: Tensor, iscale: Tensor, min_max: list
) -> tuple:
    iscale = iscale.unsqueeze(1)
    scale_rng = 2e4
    z_val = 1e-4

    W_q = c_round(W_f * iscale).clamp(min_max[0], min_max[1])

    # Build scale tensor
    W_f_c = W_f.clone()
    W_f_c_mask = torch.abs(W_f_c) < z_val
    W_f_c[W_f_c_mask] = z_val

    scale_tensor = (W_q).float() / W_f_c.float()

    # Normalize scale_tensor
    scale_b = torch.median(scale_tensor, axis=1, keepdim=True)[0]
    scale_b = scale_b.clamp(min=-scale_rng, max=scale_rng)

    # Mix with older scale
    W_r = (W_q) / scale_b
    err_b = torch.abs(W_f - W_r).mean(axis=1, keepdim=True)

    W_r = (W_q) / iscale
    err_a = torch.abs(W_f - W_r).mean(axis=1, keepdim=True)

    mask = (err_b < err_a)
    iscale_b = mask * scale_b + (~mask) * iscale

    scale_b = 1.0 / iscale_b
    qweights = (c_round(W_f * iscale_b)).clamp(min_max[0], min_max[1]).to(torch.int8) # m * n
    qweights = qweights.reshape(W_f.shape[0], -1 , 2) # m * n/2 * 2
    low_bit, high_bit = qweights.split(1, dim=-1)
    high_bit = high_bit.squeeze().view(torch.int8)
    low_bit = low_bit.squeeze().view(torch.int8)
    high_bit = high_bit << 4
    low_bit = low_bit & 0x0f
    qweights = high_bit | low_bit

    return qweights.view(torch.uint8), scale_b.to(torch.float16)
