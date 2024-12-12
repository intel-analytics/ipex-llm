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
from ipex_llm.ggml.quantize import ggml_tensor_qtype, gguf_mixed_qtype
from ..utils.common import invalidInputError
from typing import Union, Optional
import torch
from torch import nn
import logging
import numpy as np


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


_ipex_version = None


def get_ipex_version():

    global _ipex_version
    if _ipex_version is not None:
        return _ipex_version

    import intel_extension_for_pytorch as ipex
    _ipex_version = ipex.__version__
    return _ipex_version


def get_xpu_device_type(x):
    if x.device.type != "xpu":
        return x.device.type
    name = torch.xpu.get_device_name(x.device.index)
    if name.startswith("Intel(R) Arc(TM) A"):
        return "arc"
    elif name.startswith("Intel(R) Graphics [0xe20b]"):
        return "bmg"
    elif name.startswith("Intel(R) Arc(TM)"):
        if 'V' in name:
            return "lnl"
        else:
            return "mtl"
    elif name.startswith("Intel(R) Data Center GPU Flex"):
        return "flex"
    elif name.startswith("Intel(R) Data Center GPU Max"):
        return "pvc"
    elif name.startswith("Intel(R) UHD"):
        return "uhd"
    else:
        return "others"


def load_imatrix_data(imatrix_file):
    # this function is adapted from https://github.com/ggerganov/llama.cpp/blob/
    # c82d18e863fcde91b4b1109b1d0c73ea4470c405/examples/quantize/quantize.cpp#L102
    imatrix = open(imatrix_file, 'rb')
    n_entries = imatrix.read(4)
    n_entries = int.from_bytes(n_entries, 'little')
    invalidInputError(n_entries >= 1,
                      f"failed reading name for entry from {imatrix_file}")
    imatrix_data = {}
    for i in range(n_entries):
        cur_len = imatrix.read(4)
        cur_len = int.from_bytes(cur_len, 'little')
        cur_name = str(imatrix.read(cur_len), encoding='utf-8')
        # cur_name looks like blk.14.attn_output.weight for llama / mistral,
        # cur_name looks like blk.0.ffn_down.3.weight for mixtral and
        # blk.17.ffn_gate_inp.weight for mixtral
        name_list = cur_name.split('.')
        layer = name_list[1]
        module_name = name_list[2]
        exp_id = None
        if 'ffn' in module_name and len(name_list) == 4:
            module_name = module_name[4:]  # from ffn_gate to gate
        elif 'ffn' in module_name and len(name_list) == 5:
            # mixtral's mlp layer
            module_name = module_name[4:]
            exp_id = name_list[3]
        elif 'attn' in module_name:
            module_name = module_name[5]  # from attn_k to k, attn_output to o
        module_name = layer + '_' + module_name
        if exp_id is not None:
            module_name += '_' + exp_id
        ncall = imatrix.read(4)
        ncall = int.from_bytes(ncall, 'little')
        nval = imatrix.read(4)
        nval = int.from_bytes(nval, 'little')
        invalidInputError(nval >= 1,
                          f"failed reading number of values for entry {i}")
        byte_data = imatrix.read(4 * nval)
        idata = np.frombuffer(byte_data, dtype=np.float32)

        if ncall > 0:
            idata = idata / ncall
        imatrix_data[module_name] = torch.from_numpy(idata).float()

    print(f"loaded {len(imatrix_data)} importance matrix entries from {imatrix_file}.")
    return imatrix_data


def module_name_process(full_module_name):
    # full name maybe model.layers.31.self_attn.o_proj for llama/mistral
    # full name maybe model.layers.0.block_sparse_moe.gate or
    # model.layers.0.block_sparse_moe.experts.0.w1 for mixtral
    module_name_list = full_module_name.split('.')
    if len(module_name_list) >= 5:
        super_module_name = module_name_list[3]
    else:
        super_module_name = None
    exp_id = None
    new_module_name = None
    layer = None
    cur_module = None
    dq_idx = None
    if super_module_name == 'block_sparse_moe':
        # handle mixtral moe here
        moe_mapping = {"w1": "gate", "w2": "down", "w3": "up"}
        layer = module_name_list[2]
        if len(module_name_list) == 5 and module_name_list[-1] == 'gate':
            cur_module = 'gate_inp'  # mapping with imatrix
        elif len(module_name_list) == 7:
            exp_id = module_name_list[-2]
            cur_module = module_name_list[-1]
            cur_module = moe_mapping[cur_module]
        new_module_name = '_'.join([layer, cur_module])
        if exp_id is not None:
            new_module_name += '_' + exp_id
    else:
        if len(module_name_list) == 5:
            layer = module_name_list[2]
            cur_module = module_name_list[-1][:-5]
            new_module_name = '_'.join([layer, cur_module])
        elif len(module_name_list) == 6 and 'dq' in module_name_list[-1]:
            # for NPU dq_list linear
            layer = module_name_list[2]
            cur_module = module_name_list[-1]
            try:
                dq_idx = int(cur_module[-2:])
            except:
                dq_idx = int(cur_module[-1:])
            if cur_module[0] in 'qkvo':
                cur_module = cur_module[0]
            elif cur_module[:2] == "up":
                cur_module = cur_module[:2]
            elif cur_module[:4] == "gate" or cur_module[:4] == "down":
                cur_module = cur_module[:4]
            new_module_name = '_'.join([layer, cur_module])
        elif len(module_name_list) == 1:
            new_module_name = module_name_list[0]
    return new_module_name, layer, cur_module, dq_idx


def get_cur_qtype_and_imatrix(qtype, full_module_name, imatrix_data, model_config=None):
    cur_qtype = qtype
    cur_imatrix = None
    if model_config is not None:
        model_type = getattr(model_config, "model_type", None)
    else:
        model_dtype = None

    if qtype in [ggml_tensor_qtype["gguf_iq2_xxs"], ggml_tensor_qtype["gguf_iq2_xs"],
                 ggml_tensor_qtype["gguf_iq1_s"]]:
        # For quantization which needs importance matrix
        new_module_name, layer, cur_module, _ = module_name_process(full_module_name)
        # custom mixed quantization strategy
        if model_type == "mixtral":
            if cur_module == 'v':
                # llama.cpp use q4_K here
                cur_qtype = ggml_tensor_qtype['sym_int4']
            elif cur_module == 'down' and int(layer) in [0, 1, 2, 3]:
                cur_qtype = ggml_tensor_qtype['q2_k']
        else:
            num_hidden_layers = getattr(model_config, "num_hidden_layers", None)
            hidden_size = getattr(model_config, "hidden_size", None)
            if model_type == "llama" and hidden_size == 8192:
                # for llama2-70b
                if cur_module == 'v':
                    cur_qtype = ggml_tensor_qtype['sym_int4']  # llama.cpp use q4k here
                if cur_module == 'down' and int(layer) < int(num_hidden_layers/8):
                    cur_qtype = ggml_tensor_qtype['q2_k']
            elif cur_module == 'v' or (cur_module == 'down' and int(layer) in [0, 1, 10, 11]):
                cur_qtype = ggml_tensor_qtype['q2_k']
            if qtype == ggml_tensor_qtype["gguf_iq1_s"] and cur_module == 'o':
                cur_qtype = ggml_tensor_qtype['gguf_iq2_xxs']
        if imatrix_data is not None and new_module_name in imatrix_data:
            cur_imatrix = imatrix_data[new_module_name]
        else:
            # if no imatrix is available, use sym_int8 for lm_head
            cur_imatrix = None
            if new_module_name == 'lm_head':
                cur_qtype = ggml_tensor_qtype['sym_int8']
    elif qtype == ggml_tensor_qtype["q2_k"]:
        new_module_name, layer, cur_module, _ = module_name_process(full_module_name)
        if cur_module == 'v' or (cur_module == 'down' and int(layer) in [0, 1, 10, 11]):
            # TODO: q2_k need others k-quants type here
            cur_qtype = ggml_tensor_qtype['q2_k']
        if imatrix_data is not None and new_module_name in imatrix_data:
            cur_imatrix = imatrix_data[new_module_name]
        else:
            # if no imatrix is available, use sym_int8 for lm_head
            cur_imatrix = None
            if new_module_name == 'lm_head':
                cur_qtype = ggml_tensor_qtype['sym_int8']
    elif qtype > 100:
        # gguf mixed precision
        new_module_name, layer, cur_module, _ = module_name_process(full_module_name)
        num_hidden_layers = getattr(model_config, "num_hidden_layers", None)
        if qtype in [gguf_mixed_qtype["gguf_q4k_s"], gguf_mixed_qtype["gguf_q4k_m"]] and \
                new_module_name == 'lm_head':
            cur_qtype = ggml_tensor_qtype['q6_k']
        elif qtype == gguf_mixed_qtype["gguf_q4k_m"]:
            if int(layer) < int(num_hidden_layers/2) and cur_module in ['v', 'down']:
                cur_qtype = ggml_tensor_qtype['q6_k']
            else:
                cur_qtype = ggml_tensor_qtype['q4_k']
        elif qtype == gguf_mixed_qtype["gguf_q4k_s"]:
            if int(layer) < int(num_hidden_layers/8) and cur_module in ['v', 'down']:
                cur_qtype = ggml_tensor_qtype['q5_k']
            else:
                cur_qtype = ggml_tensor_qtype['q4_k']
    else:
        pass
    return cur_qtype, cur_imatrix


def get_modelscope_hf_config(model_id_or_path: str,
                             revision: Optional[str] = None):
    # Read hf config dictionary from modelscope hub or local path
    from modelscope.utils.constant import ModelFile
    from modelscope.hub.file_download import model_file_download
    from modelscope.utils.config import Config
    if not os.path.exists(model_id_or_path):
        local_path = model_file_download(
            model_id_or_path, ModelFile.CONFIG, revision=revision)
    elif os.path.isdir(model_id_or_path):
        local_path = os.path.join(model_id_or_path, ModelFile.CONFIG)
    elif os.path.isfile(model_id_or_path):
        local_path = model_id_or_path
    return Config._file2dict(local_path)


def is_torch_bf16_gpu_available():
    # always true for XPU and CPU
    return True


def check_hidden_size(qtype, hidden_size):
    if hidden_size % 256 != 0:
        if qtype == ggml_tensor_qtype["q4_k"]:
            logger.info(f"hidden size {hidden_size} is not divisible by 256, "
                        "required for q4_k - using fallback quantization asym_int4.")
            return ggml_tensor_qtype["asym_int4"]
        elif qtype == ggml_tensor_qtype["q5_k"]:
            logger.info(f"hidden size {hidden_size} is not divisible by 256, "
                        "required for q5_k - using fallback quantization asym_int5.")
            return ggml_tensor_qtype["asym_int5"]
        elif qtype == ggml_tensor_qtype["q6_k"]:
            logger.info(f"hidden size {hidden_size} is not divisible by 256, "
                        "required for q6_k - using fallback quantization sym_int8.")
            return ggml_tensor_qtype["sym_int8"]
        elif qtype == ggml_tensor_qtype["fp6_k"]:
            logger.info(f"hidden size {hidden_size} is not divisible by 256, "
                        "required for fq6_k - using fallback quantization fp6.")
            return ggml_tensor_qtype["fp6"]
    return qtype


# Arc platfrom does not support FP64,
# Disable FP64 in DeepSpeedZeroOptimizer_Stage3's _constant_buffered_norm2  method
# https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/zero/stage3.py#L1365
def _constant_buffered_norm2(self, input, buffer_size=250000000):
    norm = None
    for part in input.view(-1).split(buffer_size):
        if norm is None:
            norm = part.data.norm(2)**2.0
        else:
            norm += part.data.norm(2)**2.0
    return norm**0.5
