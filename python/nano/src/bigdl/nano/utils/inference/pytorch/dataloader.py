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

from bigdl.nano.utils.inference.pytorch.model_utils import get_forward_args
from typing import Sequence
from copy import deepcopy


def transform_multiple_input_dataloader_to_inc_mode(model, dataloader):
    need_transformation, forward_args_len = _need_dataloader_type_transformation(model, dataloader)
    if need_transformation:
        # define a decorator to change multiple inputs to 2 items
        def tuple_collate_fn_wrapper(func, forward_args_len):
            def collate_fn(batch):
                res = func(batch)
                if len(res) - forward_args_len == 1:
                    # if only one y is provided
                    return tuple(res[:forward_args_len]), res[-1]
                else:
                    # if multiple y are provided
                    return tuple(res[:forward_args_len]), tuple(res[forward_args_len:])
            return collate_fn

        # deepcopy the dataloader so that the transformation will not pollute the original one
        new_dataloader = deepcopy(dataloader)

        # add collate fn to the dataloader
        new_dataloader.collate_fn = tuple_collate_fn_wrapper(new_dataloader.collate_fn,
                                                             forward_args_len)

        return new_dataloader
    return dataloader


def _need_dataloader_type_transformation(model, dataloader):
    # get forward method's parameter number
    forward_args = get_forward_args(model)
    forward_args_len = len(forward_args)

    # if the model is a simple model(x) format
    # we don't need to transform the dataloader
    # a special case is 0, this means *args is used in
    # users' forward method, we will also skip it as well
    if forward_args_len <= 1:
        return False, forward_args_len

    # check if a dataloader has met inc format
    input_sample = next(iter(dataloader))
    if isinstance(input_sample[0], Sequence):
        if len(input_sample[0]) == forward_args_len:
            return False, forward_args_len
    return True, forward_args_len
