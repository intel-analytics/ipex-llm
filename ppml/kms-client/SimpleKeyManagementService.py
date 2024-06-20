import uuid
import base64
from cryptography.fernet import Fernet

enroll_map = {}
_appid = ''
_appkey = ''

def read_encrypted_key_file(encrypted_key_path):
    with open(encrypted_key_path, 'rb') as file:
        original = file.read()
    return original

def write_encrypted_key_file(encrypted_key_path, content):
    with open(encrypted_key_path, 'wb') as file:
        file.write(content)

def set_appid_and_appkey(appid, appkey):
    global _appid
    global _appkey
    _appid = appid
    _appkey = appkey

def enroll():
    while True:
        appid = str(uuid.uuid4().int>>64)[0:12]
        appkey = str(uuid.uuid4().int>>64)[0:12]
        if appid not in enroll_map:
            enroll_map[appid] = appkey
            set_appid_and_appkey(appid, appkey)
            return {appid, appkey}

def retrieve_primary_key(primary_key_path):
    if primary_key_path == '':
        print('[Error] primary_key_save_path should be specified')
        return
    if enroll_map[_appid] != _appkey:
        print('[Error] appid and appkey do not match!')
        return
    encrypted_primary_key = Fernet.generate_key()
    write_encrypted_key_file(primary_key_path, encrypted_primary_key)

def retrieve_data_key(primary_key_path, data_key_save_path):
    if primary_key_path == '':
        print('[Error] primary_key_path should be specified')
        return
    if  data_key_save_path == '':
        print('[Error] data_key_save_path should be specified')
        return
    if enroll_map[_appid] != _appkey:
        print('[Error] appid and appkey do not match!')
        return
    primary_key_plaintext = read_encrypted_key_file(primary_key_path)
    data_key_plaintext = Fernet.generate_key()
    fernet = Fernet(primary_key_plaintext)
    data_key_ciphertext = fernet.encrypt(data_key_plaintext)
    write_encrypted_key_file(data_key_save_path, data_key_ciphertext)

def retrieve_data_key_plaintext(primary_key_path, data_key_path):
    if primary_key_path == '':
        print('[Error] primary_key_path should be specified')
        return
    if  data_key_path == '':
        print('[Error] data_key_path should be specified')
        return
    if enroll_map[_appid] != _appkey:
        print('[Error] appid and appkey do not match!')
        return
    primary_key_ciphertext = read_encrypted_key_file(primary_key_path)
    data_key_ciphertext = read_encrypted_key_file(data_key_path)
    fernet = Fernet(primary_key_ciphertext)
    data_key_plaintext = fernet.decrypt(data_key_ciphertext)
    return data_key_plaintext

