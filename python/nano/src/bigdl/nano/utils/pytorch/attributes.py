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


import torch
from torch import nn


def patch_attrs_from_model_to_object(model: nn.Module, instance):
    """
    Patch non nn.Module public attributes of original nn.Module to a new instance.

    :param model: a torch.nn.Module
    :param instance: a instance of any object
    """
    for attr in dir(model):
        if attr not in dir(instance) and not attr.startswith('_') and not\
                isinstance(getattr(model, attr), torch.nn.Module):
            setattr(instance, attr, getattr(model, attr))
