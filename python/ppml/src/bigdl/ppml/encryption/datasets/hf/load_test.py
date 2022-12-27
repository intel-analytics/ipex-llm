# simple test, delete later
import os
from bigdl.ppml.encryption.datasets.hf.load import load_from_disk
from cryptography.fernet import Fernet
import random
import base64

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
        save_path = os.path.join(save_dir, file_name + '.encrypted')
        write_data_file(save_path, encrypted)
        print('[INFO] Encrypt Successfully! Encrypted Output Is ' + save_path)
    print('[INFO] Encrypted Files.')


def encrypt():
    encrypt_directory_automation("/ppml/save-datasets/train/", "/ppml/save-datasets-encrypted/train/")
    encrypt_directory_automation("/ppml/save-datasets/test/", "/ppml/save-datasets-encrypted/test/")
    encrypt_directory_automation("/ppml/save-datasets/valid/", "/ppml/save-datasets-encrypted/valid/")


def main():
    temp = load_from_disk("/ppml/save-datasets-encrypted/test", _create_random(32))




if __name__ == "__main__":
    encrypt()