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


_torch_encryption_patch = None
is_encryption_patched = False


def patch_encryption():
    """
    patch_torch is used to patch torch.save and torch.load methods to replace original ones.

    Patched details include:

    | 1. torch.save is now located at bigdl.nano.pytorch.encryption.save
    | 2. torch.load is now located at bigdl.nano.pytorch.encryption.load

    A key argument is added to torch.save and torch.load which is used to
    encrypt/decrypt the content before saving/loading it to/from disk.

    .. note::

       Please be noted that the key is only secured in Intel SGX mode.
    """
    global is_encryption_patched
    if is_encryption_patched:
        return
    mapping_torch = _get_encryption_patch_map()
    for mapping_iter in mapping_torch:
        setattr(mapping_iter[0], mapping_iter[1], mapping_iter[2])
    is_encryption_patched = True


def _get_encryption_patch_map():
    global _torch_encryption_patch
    import torch
    from bigdl.nano.pytorch.encryption import save, load
    _torch_encryption_patch = []
    _torch_encryption_patch += [
        [torch, "old_save", torch.save],
        [torch, "old_load", torch.load],
        [torch, "save", save],
        [torch, "load", load],
    ]
    return _torch_encryption_patch
