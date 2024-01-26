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

# This file is adapted from
# https://github.com/oobabooga/text-generation-webui/blob/main/modules/RoPE.py


def get_alpha_value(alpha, base):
    '''
    Gets alpha_value from alpha_value and rope_freq_base
    '''
    if base > 0:
        return (base/10000.) ** (63/64.)
    else:
        return alpha


def get_rope_freq_base(alpha, base):
    '''
    Gets rope_freq_base from alpha_value and rope_freq_base
    '''
    if base > 0:
        return base
    else:
        return 10000 * alpha ** (64/63.)
