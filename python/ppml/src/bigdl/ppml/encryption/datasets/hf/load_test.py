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

# simple test, delete later
import os
from bigdl.ppml.encryption.datasets.hf.load import load_from_disk
import datasets
from cryptography.fernet import Fernet
import random
import base64
import torch

def _create_random(length) -> str:
    random.seed(0)
    ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chars = []
    for i in range(length):
        chars.append(random.choice(ALPHABET))
    key = "".join(chars)
    key_bytes = key.encode("ascii")
    base64_str = base64.b64encode(key_bytes)
    print(len(base64_str), flush=True)
    return base64_str

def read_data_file(data_file_path):
    with open(data_file_path, 'rb') as file:
        original = file.read()
    return original

def write_data_file(data_file_path, content):
    with open(data_file_path, 'wb') as file:
        file.write(content)

def encrypt_directory_automation(input_dir, save_dir):
    print('[INFO] Encrypt Files Start...')
    if save_dir is None:
        if input_dir[-1] == '/':
            input_dir = input_dir[:-1]
        save_dir = input_dir + '.encrypted'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    fernet = Fernet(_create_random(32))
    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        encrypted = fernet.encrypt(read_data_file(input_path))
        save_path = os.path.join(save_dir, file_name)
        write_data_file(save_path, encrypted)
        print('[INFO] Encrypt Successfully! Encrypted Output Is ' + save_path)
    print('[INFO] Encrypted Files.')


def encrypt():
    encrypt_directory_automation("/ppml/save-datasets/train/", "/ppml/save-datasets-encrypted/train/")
    encrypt_directory_automation("/ppml/save-datasets/test/", "/ppml/save-datasets-encrypted/test/")
    encrypt_directory_automation("/ppml/save-datasets/valid/", "/ppml/save-datasets-encrypted/valid/")


def main():
    temp = load_from_disk("/ppml/save-datasets-encrypted/test", keep_in_memory=True, key=_create_random(32))
    print(temp[0])

def main2():
    temp = load_from_disk("/ppml/save-datasets/test")
    print(temp[0])
    torch.save(temp, "tempsave.pt")


def main3():
    temp_dataset = torch.load("tempsave.pt")
    print(temp_dataset[0])


def generate_datasetdict():
    train_dataset = load_from_disk("/ppml/save-datasets/train")
    test_dataset = load_from_disk("/ppml/save-datasets/test")
    dataset_dict = datasets.dataset_dict.DatasetDict({'train': train_dataset, 'test': test_dataset})
    return dataset_dict

def save_datasetdict_to_disk():
    temp = generate_datasetdict()
    temp.save_to_disk("/ppml/datasetdict")
    torch.save(temp, "datasetdict.pt")


def test():
    dc = torch.load("datasetdict.pt")
    print(dc['train'][1])


def try_load_from_disk():
    dc = load_from_disk("/ppml/datasetdict")
    print(dc)


if __name__ == "__main__":
    save_datasetdict_to_disk()