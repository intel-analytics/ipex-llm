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

import torch


def get_dummy_unet_additional_residuals():
    down_block_additional_residuals = []
    down_block_additional_residuals.extend([torch.zeros(2, 320, 64, 64)] * 3)
    down_block_additional_residuals.extend([torch.zeros(2, 320, 32, 32)])
    down_block_additional_residuals.extend([torch.zeros(2, 640, 32, 32)] * 2)
    down_block_additional_residuals.extend([torch.zeros(2, 640, 16, 16)])
    down_block_additional_residuals.extend([torch.zeros(2, 1280, 16, 16)] * 2)
    down_block_additional_residuals.extend([torch.zeros(2, 1280, 8, 8)] * 3)
    mid_block_additional_residual = torch.zeros(2, 1280, 8, 8)
    return down_block_additional_residuals, mid_block_additional_residual
