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

from Crypto.Cipher import AES
from base64 import b64decode, b64encode
from Crypto import Random

def read_data_file(data_file_path):
    with open(data_file_path, 'rb') as data_file:
        original = data_file.read()
    return original

def write_data_file(data_file_path, content):
    with open(data_file_path, 'wb') as data_file:
        data_file.write(content)

def write_bytes_as_string_to_file(data_file_path, content):
    with open(data_file_path, "w") as data_file:
        data_file.write(str(content, 'utf-8'))

class AESCBCCipher:
    def __init__(self, base64_secret_key: str):
        self.key = b64decode(base64_secret_key) # crypto key bytes
        self.mode = AES.MODE_CBC
        self.block_size = 16 # block size of AES CBC is constant
        self.str_encoding = 'utf-8'
        self.PCKS5Padding = lambda s:\
          s + (self.block_size - len(s) % self.block_size)\
          * chr(self.block_size - len(s) % self.block_size)
        self.PCKS5UnPadding = lambda s: s[:-ord(s[len(s) - 1:])]

    def update(self, IV = None):
        if IV == None:
            self.IV = Random.new().read(self.block_size)
        else:
            self.IV = IV
        self.cipher = AES.new(key=self.key, mode=self.mode, IV=self.IV)

    def encrypt_file(self, plain_text_data_file_path, encrypted_file_save_path = ""):
        if (encrypted_file_save_path == ""):
            encrypted_file_save_path = plain_text_data_file_path + ".cbc"
        else:
            if not encrypted_file_save_path.endswith(".cbc"):
                encrypted_file_save_path = encrypted_file_save_path + ".cbc"
                print("[INFO] encrypted_file_save_path is changed to "\
                    + encrypted_file_save_path)
        self.update()
        data_bytes = self.PCKS5Padding(\
                str(read_data_file(plain_text_data_file_path), self.str_encoding)\
                ).encode(self.str_encoding)
        encrypted_bytes = self.cipher.encrypt(data_bytes)
        write_data_file(encrypted_file_save_path, self.IV + encrypted_bytes)
    
    def decrypt_file(self, encrypted_data_file_path,
                    decrypted_file_save_path = "", save_as_bytes = True):
        if (decrypted_file_save_path == ""):
            decrypted_file_save_path = encrypted_data_file_path + ".decrypted"
        encrypted_bytes = read_data_file(encrypted_data_file_path)
        IV = encrypted_bytes[:self.block_size]
        self.update(IV)
        decrypted_bytes = self.PCKS5UnPadding(self.cipher.decrypt(encrypted_bytes[16:]))
        if save_as_bytes:
            write_data_file(decrypted_file_save_path, decrypted_bytes)
        else: # save as string
            write_bytes_as_string_to_file(decrypted_file_save_path, decrypted_bytes)
    
    # for small string
    def encrypt(self, plain_text_string):
        self.update()
        data_bytes = self.PCKS5Padding(plain_text_string).encode(self.str_encoding)
        encrypted_bytes = self.cipher.encrypt(data_bytes)
        return b64encode(self.IV + encrypted_bytes)

    # for small string
    def decrypt(self, encrypted_base64_string):
        encrypted_bytes = b64decode(encrypted_base64_string)
        IV = encrypted_bytes[:self.block_size]
        self.update(IV)
        decrypted_bytes = self.PCKS5UnPadding(self.cipher.decrypt(encrypted_bytes[16:]))
        return str(decrypted_bytes, self.str_encoding)


