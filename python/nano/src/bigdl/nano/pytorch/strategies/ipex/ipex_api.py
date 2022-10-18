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

from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10


def create_IPEXAccelerator(*args, **kwargs):
    """Create IPEX accelerator."""
    if TORCH_VERSION_LESS_1_10:
        from .version_1_9 import IPEXAccelerator
    else:
        from .ipex_accelerator import IPEXAccelerator
    return IPEXAccelerator(*args, **kwargs)


def create_IPEXStrategy(*args, **kwargs):
    """Create IPEX strategy."""
    if TORCH_VERSION_LESS_1_10:
        from .version_1_9 import IPEXStrategy
    else:
        from .ipex_strategy import IPEXStrategy
    return IPEXStrategy(*args, **kwargs)


def ipex_optimize(*args, **kwargs):
    """Optimize model to apply IPEX's optimizations in IPEX 1.11."""
    import intel_extension_for_pytorch as ipex
    return ipex.optimize(*args, **kwargs)


def ipex_device():
    """Return the device of IPEX when using ipex 1.9."""
    import intel_pytorch_extension as ipex
    return ipex.DEVICE


def to_cpu(*args, **kwargs):
    """Recursively move the tensor in the output to the cpu inplace."""
    from .version_1_9 import to_cpu
    return to_cpu(*args, **kwargs)
