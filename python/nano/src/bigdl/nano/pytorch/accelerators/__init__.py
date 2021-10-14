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


from .ipex_torchfunctional import apply_torch_functional_replacement
from typing import Dict, List, Tuple
import pickle
import copy
from logging import warning
import torch
_torch_save = torch.save

# To replace torch.save in ipex, you need to import and exec their __init__.py first.
# And then you can replace torch.save with your customized function.
try:
    from intel_pytorch_extension.ops.save import *
except ImportError:
    warning("IPEXAccelerator requires intel_pytorch_extension installed, \
    please run `pip install torch_ipex -f https://software.intel.com/ipex-whl-stable` \
    to get IPEX ready.")
    # process needs to stop here
    raise ImportError

# Note that you need to temporarily store original torch.save,
# because it will be modified in ipex.ops.save.
torch.save = _torch_save


RESTORE_TYPE = (torch.Tensor, Dict, List, Tuple)

DEFAULT_PROTOCOL = 2

torch_save = torch.save


def to_cpu(obj):
    # Recursively move the tensor in the output to the cpu inplace.
    if torch.is_tensor(obj):
        if obj.device.type == 'xpu':
            obj = obj.cpu()
        return obj

    if isinstance(obj, RESTORE_TYPE):
        iter_keys = obj.keys() if isinstance(obj, Dict) else range(len(obj))
        for k in iter_keys:
            if isinstance(obj[k], RESTORE_TYPE):
                obj[k] = to_cpu(obj[k])

    return obj


def nano_save(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL,
              _use_new_zipfile_serialization=False):
    # Extend original `save` defined in ipex.ops.save
    # to support converting a list of xpu tensor to cpu in torch.save
    if isinstance(obj, RESTORE_TYPE):
        obj_copy = copy.deepcopy(obj)
        obj_copy = to_cpu(obj_copy)
    elif isinstance(obj, torch.nn.Module):
        obj_copy = copy.deepcopy(obj).to('cpu')
    else:
        obj_copy = obj

    return torch_save(obj_copy, f, pickle_module, pickle_protocol, _use_new_zipfile_serialization)


torch.save = nano_save

apply_torch_functional_replacement()
