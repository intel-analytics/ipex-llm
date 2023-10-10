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
# ===========================================================================
#
# This file is adapted from
# https://github.com/ggerganov/llama.cpp/blob/master/convert.py#L516
#
# MIT License
#
# Copyright (c) 2023 Georgi Gerganov
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
from torch.serialization import StorageType
import pickle
import zipfile
import io
from typing import Dict, IO, Any, Callable, List
from dataclasses import dataclass
from .common import invalidInputError


item_size = {torch.bfloat16: 2,
             torch.float16: 2,
             torch.int: 4,
             torch.float: 4,
             torch.float32: 4,
             torch.int8: 1}


@dataclass
class LazyStorage:
    load: Callable[[int, int], torch.Tensor]
    kind: StorageType
    description: str


@dataclass
class LazyTensor:
    _load: Callable[[], torch.Tensor]
    shape: List[int]
    data_type: torch.dtype
    description: str

    def load(self) -> torch.Tensor:
        ret = self._load()
        return ret

    def to(self, data_type):
        # self.validate_conversion_to(data_type)

        def load() -> torch.Tensor:
            print(f"to {data_type}")
            return self.load().to(data_type)
        return LazyTensor(load, self.shape, data_type, f'convert({data_type}) {self.description}')


def _load(pickle_fp, map_location, picklemoudle, pickle_file='data.pkl', zip_file=None):

    load_module_mapping: Dict[str, str] = {
        'torch.tensor': 'torch._tensor'
    }

    class LazyUnpickler(picklemoudle.Unpickler):
        def __init__(self, fp: IO[bytes], data_base_path: str, zip_file: zipfile.ZipFile):
            super().__init__(fp)
            self.data_base_path = data_base_path
            self.zip_file = zip_file

        def persistent_load(self, pid):
            data_type = pid[1].dtype
            filename_stem = pid[2]
            filename = f'{self.data_base_path}/{filename_stem}'
            info = self.zip_file.getinfo(filename)

            def load(offset: int, elm_count: int):
                dtype = data_type
                fp = self.zip_file.open(info)
                fp.seek(offset * item_size[dtype])
                size = elm_count * item_size[dtype]
                data = fp.read(size)
                return torch.frombuffer(bytearray(data), dtype=dtype)
            description = f'storage data_type={data_type} ' \
                          'path-in-zip={filename} path={self.zip_file.filename}'
            return LazyStorage(load=load, kind=pid[1], description=description)

        @staticmethod
        def lazy_rebuild_tensor_v2(storage: Any,
                                   storage_offset: Any,
                                   size: Any,
                                   stride: Any,
                                   requires_grad: Any,
                                   backward_hooks: Any,
                                   metadata: Any = None) -> LazyTensor:
            invalidInputError(isinstance(storage, LazyStorage),
                              "storage should be an instance of class `LazyStorage`, "
                              f"but get {type(storage)}.")

            def load() -> torch.Tensor:
                elm_count = stride[0] * size[0]
                return storage.load(storage_offset, elm_count).reshape(size)
            description = f'pickled storage_offset={storage_offset} in {storage.description}'
            return LazyTensor(load, list(size), storage.kind.dtype, description)

        @staticmethod
        def rebuild_from_type_v2(func, new_type, args, state):
            return func(*args)

        CLASSES: dict[tuple[str, str], Any] = {
            ('torch._tensor', '_rebuild_from_type_v2'): getattr(rebuild_from_type_v2, '__func__'),
            ('torch._utils', '_rebuild_tensor_v2'): getattr(lazy_rebuild_tensor_v2, '__func__'),
            ('torch', 'Tensor'): LazyTensor,
        }

        def find_class(self, mod_name, name):
            if (mod_name, name) in self.CLASSES:
                return self.CLASSES[(mod_name, name)]
            if type(name) is str and 'Storage' in name:
                try:
                    return StorageType(name)
                except KeyError:
                    pass
            mod_name = load_module_mapping.get(mod_name, mod_name)
            return super().find_class(mod_name, name)

    unpickler = LazyUnpickler(pickle_fp,
                              data_base_path=pickle_file,
                              zip_file=zip_file)
    result = unpickler.load()

    return result


# This can only be used on huggingface transformers loaded from a zip file.
def lazyload(
    f,
    *args,
    **kwargs
):
    if isinstance(f, io.BufferedIOBase):
        fp = f
    else:
        fp = open(f, 'rb')
    zf = zipfile.ZipFile(fp)
    pickle_paths = [name for name in zf.namelist() if name.endswith('.pkl')]
    invalidInputError(len(pickle_paths) == 1,
                      "There should be only one pickle_paths found, "
                      f"but get {pickle_paths}. ")
    pickle_fp = zf.open(pickle_paths[0], 'r')
    state_dict = _load(pickle_fp, None, pickle, pickle_file=pickle_paths[0][:-4], zip_file=zf)
    fp.close()  # Otherwise on windows this may be marked as reading
    return state_dict


class LazyLoadTensors:
    def __init__(self):
        self.torch_load = torch.load

    def __enter__(self):
        torch.load = lazyload

    def __exit__(self, exc_type, exc_value, traceback):
        torch.load = self.torch_load
