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
# ===========================================================================
#
# This file is adapted from
# https://github.com/casper-hansen/AutoAWQ/tree/main
#
# @article{lin2023awq,
#   title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
#   author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song},
#   journal={arXiv},
#   year={2023}
# }
#

import torch.nn as nn

class ScaledActivation(nn.Module):
    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)
    
    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)
