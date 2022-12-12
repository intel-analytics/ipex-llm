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
import os
from logging import warning


envs_checklist = ["LD_PRELOAD", "OMP_NUM_THREADS", "KMP_AFFINITY",
                  "KMP_BLOCKTIME"]

_unset_envs = []


def _check_nano_envs():
    for k in envs_checklist:
        if not os.environ.get(k, None):
            _unset_envs.append(k)

    if len(_unset_envs):
        highlight_boundary = "\n{}\n".format("*" * 150)
        warning(f"{highlight_boundary}Nano environment variables {_unset_envs} are not set.\n"
                f"Please run `source bigdl-nano-init` to initialize them, "
                f"or you can prefix `bigdl-nano-init` to the command you run.\n"
                f"\nExample:\n"
                f"bigdl-nano-init python pytorch-lenet.py --device ipex"
                f"{highlight_boundary}")

# disable env check for now, as it does not work for tf and windows
# _check_nano_envs()

try:
    import intel_extension_for_pytorch
except ModuleNotFoundError:
    import torch
    import platform
    import warnings
    _torch_version = torch.__version__.rpartition('.')[0]
    _py_version = platform.python_version().split('.')[1]
    _link = "https://github.com/intel/intel-extension-for-pytorch#installation"
    if _torch_version == '1.10' and _py_version != '9':
        warnings.warn("If you want to use the acceleration feature of ipex, "
                      f"install it from this link: {_link}")
