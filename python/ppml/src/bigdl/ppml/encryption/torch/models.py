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
import io, os
import pathlib
from cryptography.fernet import Fernet
from typing import BinaryIO, Union, IO
import pickle

DEFAULT_PROTOCOL = 2

def _is_path(name_or_buf):
    return isinstance(name_or_buf, str) or \
        isinstance(name_or_buf, pathlib.Path)

# Attention, the use of encrypt method and decrypt method here will change the file location
class _buf_operator(object):
    def __init__(self, key):
        self.secret_key = Fernet(key) 

    def encrypt_buffer(self, buffer, encrypted_buffer):
        content = buffer.getvalue()
        encrypted_content= self.secret_key.encrypt(content)
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
            raise RuntimeError(f"Expected 'r' or 'w' in mode but got {mode}")   # noqa

def save(obj, f: Union[str, os.PathLike, BinaryIO, IO[bytes]], encryption_key: str, pickle_module = pickle, pickle_protocol=DEFAULT_PROTOCOL, _use_new_zipfile_serialization=True) -> None:
    buffer = io.BytesIO()
    encrypted_buffer = io.BytesIO()
    torch.save(obj, buffer, pickle_module=pickle_module, pickle_protocol=pickle_protocol, _use_new_zipfile_serialization=_use_new_zipfile_serialization)
    encryptor = _buf_operator(encryption_key)
    encryptor.encrypt_buffer(buffer, encrypted_buffer)
    with _open_file_or_buffer(f, 'wb') as opened_file:
        opened_file.write(encrypted_buffer.getvalue())
    buffer.close()
    encrypted_buffer.close()
    return


def load(f, decryption_key, map_location=None, pickle_module=pickle, **pickle_load_args):
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
    return torch.load(decrypted_buf, map_location=map_location, **pickle_load_args)