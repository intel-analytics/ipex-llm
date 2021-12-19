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

import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import base64
import hashlib


# For cryptography < 3.0
back_end = default_backend()


def get_private_key(secret_key, salt, key_len=128):
    """
    Generate AES required random secret/privacy key
    :param secret_key: secret key in string
    :param salt: secret slat in string
    :param key_len: key len (128 or 256)
    :return: random key
    """
    # AES key_len means key bit length, not string/char length
    # 16 for 128 and 32 for 256
    bit_len = (key_len / 128) * 16
    return hashlib.pbkdf2_hmac('SHA256', secret_key.encode(), salt.encode(), 65536, int(bit_len))


def encrypt_with_AES_CBC(plain_text, secret_key, salt, key_len=128, block_size=16):
    """
    encrypt string plain text with AES CBC
    :param plain_text: plain test in string
    :param secret_key: secret key in string
    :param salt: secret slat in string
    :param key_len: key len (128 or 256)
    :param block_size: lv size (default 16 for CBC)
    :return: cipher text in string
    """
    ct_bytes = encrypt_bytes_with_AES_CBC(plain_text.encode(),
                                          secret_key, salt, key_len, block_size)
    return base64.b64encode(ct_bytes).decode()


def decrypt_with_AES_CBC(cipher_text, secret_key, salt, key_len=128, block_size=16):
    """
    decrypt string cipher text with AES CBC
    :param cipher_text: cipher text in string
    :param secret_key: secret key in string
    :param salt: secret slat in string
    :param key_len: key len (128 or 256)
    :param block_size: lv size (default 16 for CBC)
    :return: plain test in string
    """
    plain_bytes = decrypt_bytes_with_AES_CBC(base64.b64decode(cipher_text),
                                             secret_key, salt, key_len, block_size)
    return plain_bytes.decode()


def encrypt_bytes_with_AES_CBC(plain_text_bytes, secret_key, salt, key_len=128, block_size=16):
    """
    encrypt bytes plain text with AES CBC
    :param plain_text_bytes: plain test in bytes
    :param secret_key: secret key in string
    :param salt: secret slat in string
    :param key_len: key len (128 or 256)
    :param block_size: lv size (default 16 for CBC)
    :return: cipher text in bytes
    """
    key = get_private_key(secret_key, salt, key_len)
    iv = os.urandom(block_size)
    # Align with Java AES/CBC/PKCS5PADDING
    padder = padding.PKCS7(key_len).padder()
    data = padder.update(plain_text_bytes)
    data += padder.finalize()
    # create Cipher
    encryptor = Cipher(algorithms.AES(key),
                       modes.CBC(iv), backend=back_end).encryptor()
    ct = encryptor.update(data) + encryptor.finalize()
    return iv + ct


def decrypt_bytes_with_AES_CBC(cipher_text_bytes, secret_key, salt, key_len=128, block_size=16):
    """
    decrypt bytes cipher text with AES CBC
    :param cipher_text_bytes: cipher text in bytes
    :param secret_key: secret key in string
    :param salt: secret slat in string
    :param key_len: key len (128 or 256)
    :param block_size: lv size (default 16 for CBC)
    :return: plain test in bytes
    """
    key = get_private_key(secret_key, salt, key_len)
    iv = cipher_text_bytes[:block_size]
    # create Cipher
    decryptor = Cipher(algorithms.AES(key),
                       modes.CBC(iv), backend=back_end).decryptor()
    ct = decryptor.update(cipher_text_bytes[block_size:]) + decryptor.finalize()
    # unpadding
    unpadder = padding.PKCS7(key_len).unpadder()
    ct = unpadder.update(ct)
    ct += unpadder.finalize()
    return ct


def encrypt_with_AES_GCM(plain_text, secret_key, salt, key_len=128, block_size=12):
    """
    encrypt string plain text with AES GCM
    :param plain_text: plain test in string
    :param secret_key: secret key in string
    :param salt: secret slat in string
    :param key_len: key len (128 or 256)
    :param block_size: lv size (default 12 for GCM)
    :return: cipher text in string
    """
    ct_bytes = encrypt_bytes_with_AES_GCM(plain_text.encode(),
                                          secret_key, salt, key_len, block_size)
    return base64.b64encode(ct_bytes).decode()


def decrypt_with_AES_GCM(cipher_text, secret_key, salt, key_len=128, block_size=12):
    """
    decrypt string cipher text with AES GCM
    :param cipher_text: cipher text in string
    :param secret_key: secret key in string
    :param salt: secret slat in string
    :param key_len: key len (128 or 256)
    :param block_size: lv size (default 12 for GCM)
    :return: plain test in string
    """
    plain_bytes = decrypt_bytes_with_AES_GCM(base64.b64decode(cipher_text),
                                             secret_key, salt, key_len, block_size)
    return plain_bytes.decode()


def encrypt_bytes_with_AES_GCM(plain_text_bytes, secret_key, salt, key_len=128, block_size=12):
    """
    encrypt bytes plain text with AES GCM
    :param plain_text_bytes: plain test in bytes
    :param secret_key: secret key in string
    :param salt: secret slat in string
    :param key_len: key len (128 or 256)
    :param block_size: lv size (default 12 for GCM)
    :return: cipher text in bytes
    """
    key = get_private_key(secret_key, salt, key_len)
    iv = os.urandom(block_size)
    # create Cipher
    encryptor = Cipher(algorithms.AES(key),
                       modes.GCM(iv), backend=back_end).encryptor()
    ct = encryptor.update(plain_text_bytes) + encryptor.finalize()
    return iv + ct + encryptor.tag


def decrypt_bytes_with_AES_GCM(cipher_text_bytes, secret_key, salt, key_len=128, block_size=12):
    """
    decrypt bytes cipher text with AES GCM
    :param cipher_text_bytes: cipher text in bytes
    :param secret_key: secret key in string
    :param salt: secret slat in string
    :param key_len: key len (128 or 256)
    :param block_size: lv size (default 12 for GCM)
    :return: plain test in bytes
    """
    key = get_private_key(secret_key, salt, key_len)
    tag = cipher_text_bytes[-16:]
    iv = cipher_text_bytes[:block_size]
    # create Cipher
    decryptor = Cipher(algorithms.AES(key),
                       modes.GCM(iv, tag), backend=back_end).decryptor()
    # 16 for tag
    ct = decryptor.update(cipher_text_bytes[block_size:-16]) + decryptor.finalize()
    return ct
