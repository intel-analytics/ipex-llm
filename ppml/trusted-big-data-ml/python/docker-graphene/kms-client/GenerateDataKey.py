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

# default app auth conf
appid = '202112101919'
appkey = '202112345678'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--ip', type=str, help='ip address of the ehsm_kms_server', required=True)
    parser.add_argument('-port', '--port', type=str, help='port of the ehsm_kms_server',default='3000', required=False)
    parser.add_argument('-pkp', '--pkp', type=str, help='path of the primary key storage file', required=True)
    args = parser.parse_args()
    ip = args.ip
    port = args.port
    pkp = args.pkp
    return ip, port, pkp

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

def generate_data_key(ip, port, primary_key_path):
    base_url = "http://" + ip + ":" + port + "/ehsm?Action="
    headers = {"Content-Type":"application/json"}
    with open (primary_key_path, "r") as file:
        encrypted_primary_key=file.readlines()
    params=request_params({
            "cmk_base64":encrypted_primary_key[0],
            "keylen": 32,
            "aad": "test",
        })
    generateDataKeyWithoutPlaintext_resp = requests.post(url=base_url + "GenerateDataKeyWithoutPlaintext", data=json.dumps(params), headers=headers, timeout=100)
    encrypted_data_key=json.loads(generateDataKeyWithoutPlaintext_resp.text)['result']['ciphertext_base64']
    print('Writing generated data key to local file...')
    with open("./encrypted_data_key", "w") as f:
        f.write(encrypted_data_key)
    print('Generate Data Key Successfully!')

if __name__ == "__main__":
    ip,port,pkp = get_args()
    generate_data_key(ip, port, pkp)

