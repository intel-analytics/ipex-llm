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

from typing import Sequence
from copy import deepcopy
import torch
import warnings

from bigdl.nano.utils.pytorch import get_conditional_args


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


def automatic_add_label_in_dataloader(model, dataloader, input_sample=None):
    if _check_whether_add_label(model, dataloader, input_sample):
        # need to add label automaticly
        # generate a warning for user first
        warnings.warn("After checking, it is found that your data does not contain a label item. "
                      "In order to make quantification work normally, we will automatically "
                      "generate a dummy label.")

        # define a decorator to add label
        def label_collate_fn_wrapper(func):
            def collate_fn(batch):
                res = func(batch)
                # add dummy label
                return res, torch.ones(1).long()
            return collate_fn

        # construct a new dataloader
        new_dataloader = deepcopy(dataloader)
        new_dataloader.collate_fn = label_collate_fn_wrapper(new_dataloader.collate_fn)
        return new_dataloader
    return dataloader


def _need_dataloader_type_transformation(model, dataloader):
    # get forward method's parameter number
    # forward_args = get_conditional_args(model, include="all", exclude=(bool, type(None)))
    forward_args = get_conditional_args(model, include="all")
    forward_args_len = len(forward_args)

    # if the model is a simple model(x) format
    # we don't need to transform the dataloader
    # a special case is 0, this means *args is used in
    # users' forward method, we will also skip it as well
    if forward_args_len <= 1:
        return False, forward_args_len

    # check if a dataloader has met inc format
    input_sample = next(iter(dataloader))
    if isinstance(input_sample, Sequence):
        if len(input_sample) == 2 and isinstance(input_sample[1], torch.Tensor) and \
                len(input_sample[0]) <= forward_args_len:
            return False, forward_args_len
        if len(input_sample[0]) == forward_args_len:
            return False, forward_args_len
    return True, forward_args_len


def _check_whether_add_label(model, dataloader, input_sample=None):
    '''
    This function is used to check if the dataloader(calib_data) needs
    to add a (dummy) label at last.
    '''
    # get forward method's parameter number and input sample
    forward_args = get_conditional_args(model, include="all", exclude=(bool, type(None),))
    forward_args_len = len(forward_args)
    loader_input_sample = next(iter(dataloader))

    if isinstance(loader_input_sample, torch.Tensor):
        # only one tensor provided, clearly we need a dummy label
        if forward_args_len >= 1:
            return True
    elif input_sample is not None:
        if isinstance(input_sample, torch.Tensor):
            # input_sample is a Tensor
            if len(loader_input_sample) > 1:
                return False
            return True
        if len(loader_input_sample) > len(input_sample):
            # input_sample is also a sequence
            return False
    if isinstance(loader_input_sample, Sequence):
        if isinstance(loader_input_sample[0], Sequence) and \
                len(loader_input_sample[0]) == forward_args_len:
            # this means user returns a (x1, x2, ...), y
            return False
        else:
            if len(loader_input_sample) > forward_args_len:
                # this means users dataset returns at least 1 label
                return False
            else:
                # test run to check if whole sample can be used for inference
                try:
                    model(*loader_input_sample)
                    return True
                except Exception:
                    pass
                if len(loader_input_sample) == 2:
                    try:
                        model(loader_input_sample[0])
                        return False
                    except Exception:
                        if isinstance(loader_input_sample[0], Sequence):
                            try:
                                model(*loader_input_sample[0])
                                return False
                            except Exception:
                                # input sample don't contain label
                                return True
                        else:
                            return True
    return False
