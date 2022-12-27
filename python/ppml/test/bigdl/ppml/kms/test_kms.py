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
import unittest
from bigdl.ppml.kms.client import encrypt_buffer_with_key, decrypt_buffer_with_key, generate_primary_key, generate_data_key

APPID = os.environ.get('APPID')
APIKEY = os.environ.get('APIKEY')
ehsm_ip = os.environ.get('EHSM_IP')

encrypted_primary_key_path = ""
encrypted_data_key_path= ""

ehsm_port = "9000"


class TestKMS(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if APPID == None or APIKEY is None or ehsm_ip is None:
            print("Please set environment variable APPID, APIKEY and EHSM_IP before testing")
            exit(1)
        generate_primary_key(ehsm_ip, ehsm_port)
        global encrypted_primary_key_path
        encrypted_primary_key_path = "./encrypted_primary_key"
        generate_data_key(ehsm_ip, ehsm_port, encrypted_primary_key_path, 32)
        global encrypted_data_key_path
        encrypted_data_key_path = "./encrypted_data_key"
    
    def test_encrypt_buf(self):
        buf = io.BytesIO()
        buf.write(b"HELLO WORLD")
        # saved for later
        original_content = buf.getvalue()
        # Now try to encrypt the buffer
        encrypted_buf = io.BytesIO()
        encrypt_buffer_with_key(buf, encrypted_buf, ehsm_ip, ehsm_port, encrypted_primary_key_path, encrypted_data_key_path)
        decrypted_buf = io.BytesIO()
        decrypt_buffer_with_key(encrypted_buf, decrypted_buf, ehsm_ip, ehsm_port, encrypted_primary_key_path, encrypted_data_key_path)
        decrypted_content = decrypted_buf.getvalue()
        self.assertEqual(decrypted_content, original_content)
        # close all the buffer to release memory
        buf.close()
        encrypted_buf.close()
        decrypted_buf.close()
