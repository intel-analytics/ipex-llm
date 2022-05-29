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


def create_IPEXAccelerator_1_9(*args, **kwargs):
    from bigdl.nano.deps.ipex.version_1_9.ipex_accelerator_1_9 import IPEXAccelerator
    return IPEXAccelerator(*args, **kwargs)


def ipex_optimize(*args, **kwargs):
    import intel_extension_for_pytorch as ipex
    ipex.optimize(*args, **kwargs)


def ipex_device():
    from bigdl.nano.deps.ipex.version_1_9 import DEVICE
    return DEVICE
