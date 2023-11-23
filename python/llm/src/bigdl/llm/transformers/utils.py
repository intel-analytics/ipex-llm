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
# https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
# which is licensed under the MIT license:
#
# MIT License
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
from transformers.modeling_utils import _add_variant
from ..utils.common import invalidInputError
from typing import Union
import torch
from torch import nn
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"


def extract_local_archive_file(pretrained_model_name_or_path, subfolder, variant=None):
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    if os.path.isfile(
        os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant))
    ):
        # Load from a PyTorch checkpoint
        archive_file = os.path.join(
            pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant)
        )
        return archive_file, False
    elif os.path.isfile(
        os.path.join(pretrained_model_name_or_path,
                     subfolder,
                     _add_variant(WEIGHTS_INDEX_NAME, variant))
    ):
        # Load from a sharded PyTorch checkpoint
        archive_file = os.path.join(
            pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant)
        )
        is_sharded = True
        return archive_file, is_sharded
    else:
        invalidInputError(False,
                          f"Error no file named {_add_variant(WEIGHTS_NAME, variant)}"
                          " found in directory"
                          f" {pretrained_model_name_or_path}.")


def load_state_dict(checkpoint_file: Union[str, os.PathLike]):
    try:
        return torch.load(checkpoint_file, map_location="cpu")
    except Exception as e:
        invalidInputError(False,
                          f"Unable to load weights"
                          "from pytorch checkpoint file for '{checkpoint_file}' "
                          f"at '{checkpoint_file}'. ")


# PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
# so we need to apply the function recursively.
def load(module: nn.Module, state_dict, prefix=""):
    args = (state_dict, prefix, {}, True, [], [], [])
    # Parameters of module and children will start with prefix.
    # We can exit early if there are none in this state_dict
    if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            module._load_from_state_dict(*args)

    for name, child in module._modules.items():
        if child is not None:
            load(child, state_dict, prefix + name + ".")


def get_local_shard_files(pretrained_model_name_or_path, index_filename, subfolder=""):
    import json

    invalidInputError(os.path.isfile(index_filename),
                      "Can't find a checkpoint index"
                      f" ({index_filename}) in {pretrained_model_name_or_path}.")

    with open(index_filename, "r") as f:
        index = json.loads(f.read())

    shard_filenames = sorted(set(index["weight_map"].values()))
    sharded_metadata = index["metadata"]
    sharded_metadata["all_checkpoint_keys"] = list(index["weight_map"].keys())
    sharded_metadata["weight_map"] = index["weight_map"].copy()

    shard_filenames = [os.path.join(pretrained_model_name_or_path, subfolder, f)
                       for f in shard_filenames]
    return shard_filenames, sharded_metadata


def fix_key(key):
    if "beta" in key:
        return key.replace("beta", "bias")
    if "gamma" in key:
        return key.replace("gamma", "weight")
    return key


def get_autocast_dtype(x):
    if x.device.type == "xpu":
        if torch.xpu.is_autocast_xpu_enabled():
            return torch.xpu.get_autocast_xpu_dtype()
        else:
            return None
    elif x.device.type == "cpu":
        if torch.is_autocast_cpu_enabled():
            return torch.get_autocast_cpu_dtype()
        else:
            return None
    else:
        invalidInputError(False,
                          f"Device {x.device} is not supported.")
