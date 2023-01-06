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
import io
import os
import pathlib
from cryptography.fernet import Fernet
from typing import BinaryIO, Union, IO, Optional
import pickle
from bigdl.nano.utils.log4Error import invalidInputError

DEFAULT_PROTOCOL = 2


def _is_path(name_or_buf):
    return isinstance(name_or_buf, str) or \
        isinstance(name_or_buf, pathlib.Path)


class _buf_operator(object):
    def __init__(self, key):
        self.secret_key = Fernet(key)

    def encrypt_buffer(self, buffer, encrypted_buffer):
        content = buffer.getvalue()
        encrypted_content = self.secret_key.encrypt(content)
        encrypted_buffer.write(encrypted_content)

    def decrypt_buffer(self, buffer, decrypted_buffer):
        decrypted_content = self.secret_key.decrypt(buffer.getvalue())
        decrypted_buffer.write(decrypted_content)


class _opener(object):
    def __init__(self, file_like):
        self.file_like = file_like

    def __enter__(self):
        return self.file_like

    def __exit__(self, *args):
        pass


class _open_file(_opener):
    def __init__(self, name, mode):
        super(_open_file, self).__init__(open(name, mode))

    def __exit__(self, *args):
        # Flush is automatically done when closing the file
        self.file_like.close()


class _open_buffer_reader(_opener):
    def __init__(self, buffer):
        super(_open_buffer_reader, self).__init__(buffer)


class _open_buffer_writer(_opener):
    def __exit__(self, *args):
        self.file_like.flush()


def _open_file_or_buffer(file_like, mode):
    if _is_path(file_like):
        return _open_file(file_like, mode)
    else:
        if 'w' in mode:
            return _open_buffer_writer(file_like)
        elif 'r' in mode:
            return _open_buffer_reader(file_like)
        else:
            invalidInputError(f"Expected 'r' or 'w' in mode but got {mode}")


def save(obj, f: Union[str, os.PathLike, BinaryIO, IO[bytes]],
         pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL,
         _use_new_zipfile_serialization=True, encryption_key: Optional[str] = None) -> None:
    """
    Save method that will be used to replace torch.save.

    Only used in patched torch.

    :param encryption_key: Optional[str] = None. If set to None, the original
           torch.save will be invoked.  Otherwise, the content will be encrypted
           before writing it to final destination.
    """
    if encryption_key is None:
        return torch.old_save(obj, f, pickle_module=pickle_module, pickle_protocol=pickle_protocol,
                              _use_new_zipfile_serialization=_use_new_zipfile_serialization)
    buffer = io.BytesIO()
    encrypted_buffer = io.BytesIO()
    torch.old_save(obj, buffer, pickle_module=pickle_module, pickle_protocol=pickle_protocol,
                   _use_new_zipfile_serialization=_use_new_zipfile_serialization)
    encryptor = _buf_operator(encryption_key)
    encryptor.encrypt_buffer(buffer, encrypted_buffer)
    with _open_file_or_buffer(f, 'wb') as opened_file:
        opened_file.write(encrypted_buffer.getvalue())
    buffer.close()
    encrypted_buffer.close()
    return


def load(f, map_location=None, pickle_module=pickle,
         decryption_key: Optional[str] = None, **pickle_load_args):
    """
    Load method that will be used to replace torch.load.

    Only used in patched torch.

    :param decryption_key: Optional[str] = None. If set to None, the original
           torch.load will be invoked.  Otherwise, the content will be decrypted
           before loading it back to memory.
    """
    if decryption_key is None:
        return torch.old_load(f, pickle_module=pickle_module,
                              map_location=map_location, **pickle_load_args)
    decrypted_buf = io.BytesIO()
    decryptor = _buf_operator(decryption_key)
    with _open_file_or_buffer(f, 'rb') as opened_file:
        if _is_path(f):
            buf = io.BytesIO(opened_file.read())
            decryptor.decrypt_buffer(buf, decrypted_buf)
            buf.close()
        else:
            decryptor.decrypt_buffer(f, decrypted_buf)
    # After writing to the buffer, need to set it back to its original position
    decrypted_buf.seek(0)
    return torch.old_load(decrypted_buf, pickle_module=pickle_module,
                          map_location=map_location, **pickle_load_args)
