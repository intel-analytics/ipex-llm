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
from typing import Any, List, Optional, Union
from bigdl.nano.utils.common import invalidInputError


CREATE_TENSOR_FUNC = ['rand', 'randint', 'randn', 'zeros', 'ones', 'empty', 'full',
                      'rand_like', 'randint_like', 'randn_like', 'zeros_like',
                      'ones_like', 'empty_like', 'full_like',
                      'tensor', 'scalar_tensor', 'sparse_coo_tensor', 'sparse_csr_tensor',
                      'sparse_csc_tensor', 'sparse_bsc_tensor', 'sparse_bsr_tensor',
                      'sparse_compressed_tensor', 'nested_tensor'
                      'randperm', 'normal', 'range', 'arange', 'eye',
                      'as_tensor', 'asarray',
                      'linspace', 'logspace',
                      'tril_indices', 'triu_indices',
                      'bartlett_window', 'blackman_window', 'hamming_window',
                      'hann_window', 'kaiser_window',
                      'empty_quantized', 'empty_strided',
                      'frombuffer', 'from_file']


STR_TO_DTYPE = {'fp32': torch.float32,
                'float32': torch.float32,
                'fp64': torch.float64,
                'float64': torch.float64,
                'bf16': torch.bfloat16,
                'bfloat16': torch.bfloat16,
                'fp16': torch.float16,
                'float16': torch.float16}


def create_tensor_func(torch_create_tensor_func, from_dtype, to_dtype):
    def new_create_tensor_func(*args, **kwargs):
        if 'dtype' in kwargs and kwargs['dtype'] is not None:
            if kwargs['dtype'] == from_dtype:
                kwargs['dtype'] = to_dtype
        return torch_create_tensor_func(*args, **kwargs)
    return new_create_tensor_func


def np_op_func(self, *args, **kwargs):
    return self


def replace_attr(obj, name: str, value):
    torch_attr = getattr(obj, name)
    setattr(obj, name, value)


def patch_dtype(from_dtype: Union[str, torch.dtype] = "fp64",
                to_dtype: Union[str, torch.dtype] = "fp32"):

    '''
    patch_dtype is used to change the tensor's dtype in users' application
    from `from_dtype` to `to_dtype`.

    e.g.
        >>> from bigdl.nano.pytorch.patching import patch_dtype
        >>> patch_dtype(from_dtype="fp64", to_dtype="fp32")
        >>> # will replace all tensors that has fp64 precision to fp32.

    :param from_dtype: the tensors' dtype to be replaced. default to "fp64"
    :param to_dtype: the tensors' dtype to use. default to "fp32"
    '''

    if isinstance(from_dtype, str):
        invalidInputError(from_dtype.lower() in STR_TO_DTYPE.keys(),
                          f"from_dtype should be one of {STR_TO_DTYPE.keys()}, "
                          f"while get {from_dtype}.")
        from_dtype = STR_TO_DTYPE[from_dtype.lower()]

    if isinstance(to_dtype, str):
        invalidInputError(to_dtype.lower() in STR_TO_DTYPE.keys(),
                          f"to_dtype should be one of {STR_TO_DTYPE.keys()}, "
                          f"while get {to_dtype}.")
        to_dtype = STR_TO_DTYPE[to_dtype.lower()]

    # set default dtype
    torch.set_default_dtype(to_dtype)

    # patch tensor create functions
    for f in CREATE_TENSOR_FUNC:
        try:
            replace_attr(torch, f, create_tensor_func(getattr(torch, f),
                                                      from_dtype,
                                                      to_dtype))
        except AttributeError:
            pass

    # patch Tensor.double
    # TODO: add others
    if from_dtype == torch.float64:
        replace_attr(torch.Tensor, "double", np_op_func)
