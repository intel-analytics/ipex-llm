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

from restcaller import request_parimary_key_ciphertext,request_data_key_ciphertext, request_data_key_plaintext

def read_encrypted_key_file(encrypted_key_path):
    with open(encrypted_key_path, 'r') as file:
        original = file.readlines()
    return original[0]

def write_encrypted_key_file(encrypted_key_path, content):
    with open(encrypted_key_path, 'w') as file:
        file.write(content)

def retrieve_data_key_plaintext(ip, port, encrypted_primary_key_path, encrypted_data_key_path):
    encrypted_primary_key = read_encrypted_key_file(encrypted_primary_key_path)
    encrypted_data_key = read_encrypted_key_file(encrypted_data_key_path)
    data_key_plaintext = request_data_key_plaintext(ip, port, encrypted_primary_key, encrypted_data_key)
    return data_key_plaintext


def generate_primary_key_ciphertext(ip, port):
    primary_key_ciphertext = request_parimary_key_ciphertext(ip, port)
    write_encrypted_key_file('./encrypted_primary_key', primary_key_ciphertext)
    print('[INFO] Primary Key Generated Successfully at ./encrypted_primary_key')


def generate_data_key_ciphertext(ip, port, encrypted_primary_key_path, data_key_length = 32):
    encrypted_primary_key=read_encrypted_key_file(encrypted_primary_key_path)
    data_key_ciphertext = request_data_key_ciphertext(ip, port, encrypted_primary_key, data_key_length)
    write_encrypted_key_file('./encrypted_data_key', data_key_ciphertext)
    print('[INFO] Data Key Generated Successfully at ./encrypted_data_key')
