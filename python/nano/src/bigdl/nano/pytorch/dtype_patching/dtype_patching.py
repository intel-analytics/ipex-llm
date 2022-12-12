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


def create_tensor_func(torch_create_tensor_func, original_dtype, target_dtype):
    def new_create_tensor_func(*args, **kwargs):
        if 'dtype' in kwargs and kwargs['dtype'] is not None:
            if kwargs['dtype'] == original_dtype:
                kwargs['dtype'] = target_dtype
        return torch_create_tensor_func(*args, **kwargs)
    return new_create_tensor_func


def np_op_func(self, *args, **kwargs):
    return self


def replace_attr(obj, name: str, value):
    torch_attr = getattr(obj, name)
    setattr(obj, name, value)


def patch_dtype(original_dtype=torch.float64, target_dtype=torch.float32):
    # set default dtype
    torch.set_default_dtype(target_dtype)

    # patch tensor create functions
    for f in CREATE_TENSOR_FUNC:
        try:
            replace_attr(torch, f, create_tensor_func(getattr(torch, f),
                                                      original_dtype,
                                                      target_dtype))
        except AttributeError:
            pass

    # patch Tensor.float64
    # TODO: add others
    if original_dtype == torch.float64:
        replace_attr(torch.Tensor, "double", np_op_func)
