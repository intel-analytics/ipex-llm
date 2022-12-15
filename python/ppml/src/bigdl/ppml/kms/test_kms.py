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

# simple tests for kms, require pktest, just for dev
# python3 -m pktest test_kms.py

import io, os
from kms.client import encrypt_buf_with_key, decrypt_buf_with_key

# Only for test purpose, never use it in production
os.environ['APPID'] = "63a88858-29f6-426f-b9b7-15702bf056ac"
os.environ['APIKEY'] = "PxiX0hduXAG76cw1JMYPJWyGBMGc0muB"
encrypted_primary_key_path = "/ppml/encrypted_primary_key"
encrypted_data_key_path= "/ppml/encrypted_data_key"
ehsm_ip = "172.168.0.226"
ehsm_port = "9000"

def test_encrypt_buf():
    buf = io.BytesIO()
    buf.write(b"HELLO WORLD")
    # saved for later
    original_content = buf.getvalue()
    # Now try to encrypt the buffer
    encrypted_buf = io.BytesIO()
    encrypt_buf_with_key(buf, encrypted_buf, ehsm_ip, ehsm_port, encrypted_primary_key_path, encrypted_data_key_path)
    decrypted_buf = io.BytesIO()
    decrypt_buf_with_key(encrypted_buf, decrypted_buf, ehsm_ip, ehsm_port, encrypted_primary_key_path, encrypted_data_key_path)
    decrypted_content = decrypted_buf.getvalue()
    assert decrypted_content == original_content
    # close all the buffer to release memory
    buf.close()
    encrypted_buf.close()
    decrypted_buf.close()
