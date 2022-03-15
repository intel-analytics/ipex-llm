import requests
import json
import base64
import time
import random
import hmac
from hashlib import sha256
import os
from collections import OrderedDict
import urllib.parse

# default app auth conf
appid = os.getenv('APPID')
apikey = os.getenv('APIKEY')
headers = {"Content-Type":"application/json"}

def request_params(payload):
    params = OrderedDict()
    params["appid"] = appid
    ord_payload = OrderedDict(sorted(payload.items(), key=lambda k: k[0]))
    params["payload"] = urllib.parse.unquote(urllib.parse.urlencode(ord_payload))
    params["timestamp"] = str(int(time.time()*1000))
    sign_string = urllib.parse.unquote(urllib.parse.urlencode(params))
    sign = str(base64.b64encode(hmac.new(apikey.encode('utf-8'),
                                         sign_string.encode('utf-8'),
                                         digestmod=sha256).digest()),'utf-8').upper()
    params["payload"] = ord_payload
    params["sign"] = sign
    return params

def construct_url(ip, port, action):
    return "http://" + ip + ":" + port + "/ehsm?Action=" + action

def post_request(ip, port, action, payload):
    url = construct_url(ip, port, action)
    params = request_params(payload)
    create_resp = requests.post(url=url, data=json.dumps(params), headers=headers, timeout=100)
    result = json.loads(create_resp.text)['result']
    return result

def request_parimary_key_ciphertext(ip, port):
    action = "CreateKey"
    payload = {
        "keyspec":"EH_AES_GCM_128",
        "origin":"EH_INTERNAL_KEY"
    }
    primary_key_ciphertext = post_request(ip, port, action, payload)['cmk_base64']
    return primary_key_ciphertext

def request_data_key_ciphertext(ip, port, encrypted_primary_key):
    action = "GenerateDataKeyWithoutPlaintext"
    payload = {
        "cmk_base64":encrypted_primary_key,
        "keylen": 32,
        "aad": "test",
    }
    data_key_ciphertext = post_request(ip, port, action, payload)['ciphertext_base64']
    return data_key_ciphertext

def request_data_key_plaintext(ip, port, encrypted_primary_key, encrypted_data_key):
    action = "Decrypt"
    payload = {
        "cmk_base64":encrypted_primary_key,
        "ciphertext":encrypted_data_key,
        "aad":"test",
    }
    data_key_plaintext = post_request(ip, port, action, payload)['plaintext_base64']
    return data_key_plaintext
