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

def create_IPEXAccelerator(*args, **kwargs):
    from bigdl.nano.deps.ipex.ipex_accelerator import IPEXAccelerator
    return IPEXAccelerator(*args, **kwargs)


def ipex_device():
    ipex_device = 'xpu:0' # DeprecationWarning after ipex 1.9.0
    return ipex_device

def ipex_optimize(*args, **kwargs):
    from intel_extension_for_pytorch import optimize
    return optimize(*args, **kwargs)
