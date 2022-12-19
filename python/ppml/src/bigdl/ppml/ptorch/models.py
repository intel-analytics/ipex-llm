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
from . fileutils import is_path
import torch
import io

# Fixme: these arguments, should we move it to somewhere else?
# TODO: Did a through test, ensure that we can load the same arguments back.
def save(obj, f, kms_ip, kms_port, kms_encrypted_primary_key, kms_encrypted_data_key, encrypted=True):
    if encrypted==False:
        torch.save(obj, f)
        return
    buffer = io.BytesIO()
    encrypted_buf = io.BytesIO()
    torch.save(obj, buffer)
    # try to encrypt the buffer
    encrypt_buf_with_key(buffer, encrypted_buf, kms_ip, kms_port, kms_encrypted_primary_key, kms_encrypted_data_key)
    # Based on the type of f do different things.
    # f can be of type: Union[str, os.PathLike, BinaryIO, IO[bytes]]
    # 1. classify path or buffer?
    if is_path(f):
        # its a file
        # Let's just open the file with open method
        with open(f, 'wb') as opened_file:
            opened_file.write(encrypted_buf.getvalue())
            opened_file.flush()
            opened_file.close()
    else:
        # its a buffer, and we don't need to do anything
        f.write(encrypted_buf.getvalue())
        f.flush()
    
    # clean all teh buffers
    buffer.close()
    encrypted_buf.close()
    return


# TODO: do we need to move these variables to other places?
def load(f, kms_ip, kms_port, kms_encrypted_primary_key, kms_encrypted_data_key, map_location=None):
    # TODO: check seekable
    decrypted_buf = io.BytesIO()
    if is_path(f):
        with open(f, 'rb') as opened_file:
           buf = io.BytesIO(opened_file.read()) 
           decrypt_buf_with_key(buf, decrypted_buf, kms_ip, kms_port, kms_encrypted_primary_key, kms_encrypted_data_key)
    else:
        decrypt_buf_with_key(f, decrypted_buf, kms_ip, kms_port, kms_encrypted_primary_key, kms_encrypted_data_key)
    # After writing to the buffer, need to set it back to its original position
    decrypted_buf.seek(0)
    # now its in the decrypted_buf
    return torch.load(decrypted_buf, map_location=map_location)
