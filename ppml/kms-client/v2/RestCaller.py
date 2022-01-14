import requests
import json
import base64
import time
import random
import hmac
from hashlib import sha256
import os
from collections import OrderedDict

# default app auth conf
appid = '202112101919'
appkey = '202112345678'
headers = {"Content-Type":"application/json"}

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

def construct_url(ip, port, action):
    return "http://" + ip + ":" + port + "/ehsm?Action=" + action

def post_request(ip, port, action, payload):
    url = construct_url(ip, port, action)
    params = request_params(payload)
    create_resp = requests.post(url=url, data=json.dumps(params), headers=headers, timeout=100)
    key = json.loads(create_resp.text)['result']['cmk_base64']
    return key

def request_parimary_key_ciphertext(ip, port):
    action = "CreateKey"
    payload = {
        "keyspec":"EH_AES_GCM_128",
        "origin":"EH_INTERNAL_KEY"
    }
    primary_key_ciphertext = post_request(ip, port, action, payload)
    return primary_key_ciphertext

def request_data_key_ciphertext(ip, port, encrypted_primary_key):
    action = "GenerateDataKeyWithoutPlaintext"
    payload = {
        "cmk_base64":encrypted_primary_key,
        "keylen": 32,
        "aad": "test",
    }
    data_key_ciphertext = post_request(ip, port, action, payload)
    return data_key_ciphertext

def request_data_key_plaintext(ip, port, encrypted_primary_key, encrypted_data_key):
    action = "Decrypt"
    payload = {
        "cmk_base64":encrypted_primary_key,
        "ciphertext":encrypted_data_key,
        "aad":"test",
    }
    data_key_plaintext = post_request(ip, port, action, payload)
    return data_key_plaintext
