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

from kms.client import encrypt_buf_with_key, decrypt_buf_with_key
import torch
import io
import pathlib


def _is_path(name_or_buf):
    return isinstance(name_or_buf, str) or \
        isinstance(name_or_buf, pathlib.Path)

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
            raise RuntimeError(f"Expected 'r' or 'w' in mode but got {mode}")


# TODO: these arguments, should we move it to somewhere else?
def save(obj, f, kms_ip, kms_port, kms_encrypted_primary_key, kms_encrypted_data_key, encrypted=True):
    if encrypted==False:
        torch.save(obj, f)
        return
    buffer = io.BytesIO()
    encrypted_buf = io.BytesIO()
    torch.save(obj, buffer)
    # Encrypt the buffer
    encrypt_buf_with_key(buffer, encrypted_buf, kms_ip, kms_port, kms_encrypted_primary_key, kms_encrypted_data_key)
    with _open_file_or_buffer(f, 'wb') as opened_file:
        opened_file.write(encrypted_buf.getvalue())
    buffer.close()
    encrypted_buf.close()
    return


# TODO: do we need to move these variables to other places?
def load(f, kms_ip, kms_port, kms_encrypted_primary_key, kms_encrypted_data_key, map_location=None):
    decrypted_buf = io.BytesIO()
    with _open_file_or_buffer(f, 'rb') as opened_file:
        if _is_path(f):
            buf = io.BytesIO(opened_file.read())
            decrypt_buf_with_key(buf, decrypted_buf, kms_ip, kms_port, kms_encrypted_primary_key, kms_encrypted_data_key)
        else:
            decrypt_buf_with_key(f, decrypted_buf, kms_ip, kms_port, kms_encrypted_primary_key, kms_encrypted_data_key)
    # After writing to the buffer, need to set it back to its original position
    decrypted_buf.seek(0)
    return torch.load(decrypted_buf, map_location=map_location)
