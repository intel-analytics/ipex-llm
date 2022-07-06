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

import operator
from pytorch_lightning.utilities.imports import _compare_version

if _compare_version("torch", operator.lt, "1.10"):
    import torch
    from torch.nn import functional as F
    from torch import Tensor
    from typing import Callable, Dict, List, Tuple
    import pickle
    import copy
    from logging import warning
    from bigdl.nano.utils.log4Error import invalidInputError

    _cross_entropy = F.cross_entropy
    _torch_save = torch.save

    # To replace torch.save in ipex, you need to import and exec their __init__.py first.
    # And then you can replace torch.save with your customized function.
    try:
        from intel_pytorch_extension.ops.save import *
    except ImportError:
        msg = "IPEXAccelerator requires intel_pytorch_extension installed, \
        please run `pip install torch_ipex -f https://software.intel.com/ipex-whl-stable` \
        to get IPEX ready."
        warning(msg)
        # process needs to stop here
        invalidInputError(False, msg)

    def _replace_torch_function(function_name: str, replace_func: Callable):
        setattr(torch.nn.functional, function_name, replace_func)

    def _workaround_cross_entropy(
        input: Tensor,
        target: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        input = input.cpu()
        target = target.cpu()
        return _cross_entropy(input, target, *args, **kwargs)

    # For the development of ipex is on going, there are some
    # operations or functions we can not call directly like:
    # https://github.com/intel-analytics/analytics-zoo/pull/4600#discussion_r699773873
    #
    # Sometimes there are workarounds like moving tensor to cpu or following other implementation,
    # so we can replace `torch.nn.functional.SOMEFUNCTION` with our roundabout method.
    # Like `workaround_cross_entropy`, these method must has the same signature as the origin.
    #
    # The replacement only takes place when ipex accelerator is imported,
    # in another word `use_ipex` is specfied as true in Trainer, so it will not
    #  affect default behaviors.
    #
    # Usage: append your target method and your own implements to  `replacement_dict`
    #
    # Apply ops replacements here
    replacement_dict = {
        "cross_entropy": _workaround_cross_entropy
    }

    def _apply_torch_functional_replacement():
        for k, v in replacement_dict.items():
            _replace_torch_function(k, v)

    # Note that you need to temporarily store original torch.save,
    # because it will be modified in ipex.ops.save.
    torch.save = _torch_save

    RESTORE_TYPE = (torch.Tensor, Dict, List, Tuple)

    DEFAULT_PROTOCOL = 2

    torch_save = torch.save

    def to_cpu(obj):
        """Recursively move the tensor in the output to the cpu inplace."""
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
        """
        Replace torch.save.

        Extend original `save` defined in ipex.ops.save
        to support converting a list of xpu tensor to cpu in torch.save.
        """
        if isinstance(obj, RESTORE_TYPE):
            obj_copy = copy.deepcopy(obj)
            obj_copy = to_cpu(obj_copy)
        elif isinstance(obj, torch.nn.Module):
            obj_copy = copy.deepcopy(obj).to('cpu')
        else:
            obj_copy = obj

        return torch_save(obj_copy, f, pickle_module, pickle_protocol,
                          _use_new_zipfile_serialization)

    torch.save = nano_save

    _apply_torch_functional_replacement()
