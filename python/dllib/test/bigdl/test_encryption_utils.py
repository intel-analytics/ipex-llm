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
from unittest import TestCase

import pytest
import string
import random

from bigdl.dllib.utils.encryption_utils import *


class TestEncryption(TestCase):

    def setUp(self) -> None:
        # self.random_str = "hello world, hello scala, hello encrypt, come on UNITED!!!"
        letters = string.ascii_lowercase
        self.random_str = ''.join(random.choice(letters) for _ in range(100))

    def test_aes128_cbc_bytes(self):
        enc_bytes = encrypt_bytes_with_AES_CBC(self.random_str.encode(),
                                               'analytics-zoo', 'intel-analytics')
        dec_bytes = decrypt_bytes_with_AES_CBC(enc_bytes,
                                               'analytics-zoo', 'intel-analytics')
        assert dec_bytes == self.random_str.encode(), \
            "Check AES CBC 128 encryption and decryption result"

    def test_aes256_cbc_bytes(self):
        enc_bytes = encrypt_bytes_with_AES_CBC(self.random_str.encode("utf-8"),
                                               'analytics-zoo', 'intel-analytics', 256)
        dec_bytes = decrypt_bytes_with_AES_CBC(enc_bytes,
                                               'analytics-zoo', 'intel-analytics', 256)
        assert dec_bytes == self.random_str.encode("utf-8"), \
            "Check AES CBC 256 encryption and decryption result"

    def test_aes128_gcm_bytes(self):
        enc_bytes = encrypt_bytes_with_AES_GCM(self.random_str.encode(),
                                               'analytics-zoo', 'intel-analytics')
        dec_bytes = decrypt_bytes_with_AES_GCM(enc_bytes,
                                               'analytics-zoo', 'intel-analytics')
        assert dec_bytes == self.random_str.encode(), \
            "Check AES GCM 128 encryption and decryption result"

    def test_aes256_gcm_bytes(self):
        enc_bytes = encrypt_bytes_with_AES_GCM(self.random_str.encode(),
                                               'analytics-zoo', 'intel-analytics', 256)
        dec_bytes = decrypt_bytes_with_AES_GCM(enc_bytes,
                                               'analytics-zoo', 'intel-analytics', 256)
        assert dec_bytes == self.random_str.encode(), \
            "Check AES GCM 128 encryption and decryption result"

    def test_aes128_cbc(self):
        enc_str = encrypt_with_AES_CBC(self.random_str, 'analytics-zoo', 'intel-analytics')
        dec_str = decrypt_with_AES_CBC(enc_str, 'analytics-zoo', 'intel-analytics')
        assert dec_str == self.random_str, \
            "Check AES CBC 128 encryption and decryption result"

    def test_aes256_cbc(self):
        enc_str = encrypt_with_AES_CBC(self.random_str, 'analytics-zoo', 'intel-analytics', 256)
        dec_str = decrypt_with_AES_CBC(enc_str, 'analytics-zoo', 'intel-analytics', 256)
        assert dec_str == self.random_str, \
            "Check AES CBC 128 encryption and decryption result"

    def test_aes128_gcm(self):
        enc_str = encrypt_with_AES_GCM(self.random_str, 'analytics-zoo', 'intel-analytics')
        dec_str = decrypt_with_AES_GCM(enc_str, 'analytics-zoo', 'intel-analytics')
        assert dec_str == self.random_str, \
            "Check AES GCM 128 encryption and decryption result"

    def test_aes256_gcm(self):
        enc_str = encrypt_with_AES_GCM(self.random_str, 'analytics-zoo', 'intel-analytics', 256)
        dec_str = decrypt_with_AES_GCM(enc_str, 'analytics-zoo', 'intel-analytics', 256)
        assert dec_str == self.random_str, \
            "Check AES GCM 128 encryption and decryption result"


if __name__ == "__main__":
    pytest.main([__file__])
