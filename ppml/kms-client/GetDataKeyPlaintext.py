import requests
import json
import argparse
import base64
import time
import random
import hmac
from hashlib import sha256
import os
from collections import OrderedDict
import urllib.parse
from cryptography.fernet import Fernet

# default app auth conf
appid = '202112101919'
appkey = '202112345678'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--ip', type=str, help='ip address of the ehsm_kms_server', required=True)
    parser.add_argument('-port', '--port', type=str, help='port of the ehsm_kms_server',default='3000', required=False)
    parser.add_argument('-pkp', '--pkp', type=str, help='path of the primary key storage file', required=True)
    parser.add_argument('-dkp', '--dkp', type=str, help='path of the data key storage file', required=True)
    args = parser.parse_args()
    ip = args.ip
    port = args.port
    pkp = args.pkp
    dkp = args.dkp
    return ip, port, pkp, dkp

def request_params(payload):
    params = OrderedDict()

    params["appid"] = appid
    params["nonce"] = str(int.from_bytes(os.urandom(16), "big"))
    params["timestamp"] = int(time.time())

    query_string = urllib.parse.urlencode(params)
    query_string += '&app_key=' + appkey
    sign = str(base64.b64encode(hmac.new(appkey.encode('utf-8'), query_string.encode('utf-8'), digestmod=sha256).digest()),'utf-8')

    params["sign"] = sign
    params["payload"] = payload
    return params

def decrypt_data_key(ip, port, primary_key_path, data_key_path):
    base_url = "http://" + ip + ":" + port + "/ehsm?Action="
    headers = {"Content-Type":"application/json"}
    encrypted_primary_key, encrypted_data_key = None, None
    with open (primary_key_path, "r") as file:
        encrypted_primary_key=file.readlines()
    with open (data_key_path, "r") as file:
        encrypted_data_key=file.readlines()
    params=request_params({
            "cmk_base64":encrypted_primary_key[0],
            "ciphertext":encrypted_data_key[0],
            "aad":"test",
        })
    decrypt_res = requests.post(url=base_url + "Decrypt", data=json.dumps(params), headers=headers, timeout=100)
    decrypted_data_key = json.loads(decrypt_res.text)['result']['plaintext_base64']
    return decrypted_data_key

if __name__ == "__main__":
    ip,port,pkp,dkp = get_args()
    data_key = decrypt_data_key(ip, port, pkp, dkp)
    print(data_key)
