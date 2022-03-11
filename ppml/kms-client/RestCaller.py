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
appid = '33b7e68b-02f7-4393-b373-f2442350f11e'
apikey = 'PyXugvK3Eq5Sn7EgCQj3Eat1psE8HRXe'
headers = {"Content-Type":"application/json"}

def request_params(payload):
    params = OrderedDict()
    params["appid"] = appid    
    ord_payload = OrderedDict(sorted(payload.items(), key=lambda k: k[0]))
    params["payload"] = urllib.parse.unquote(urllib.parse.urlencode(ord_payload))
    params["timestamp"] = str(int(time.time()*1000))
    sign_string = urllib.parse.unquote(urllib.parse.urlencode(params))
    print(sign_string.encode('utf-8'))
    print(apikey.encode('utf-8'))
    sign = str(base64.b64encode(hmac.new(apikey.encode('utf-8'), sign_string.encode('utf-8'), digestmod=sha256).digest()),'utf-8').upper()
    params["payload"] = ord_payload
    params["sign"] = sign
    return params

def construct_url(ip, port, action):
    return "http://" + ip + ":" + port + "/ehsm?Action=" + action

def post_request(ip, port, action, payload):
    url = construct_url(ip, port, action)
    params = request_params(payload)
    print("request params is: ", params)
    create_resp = requests.post(url=url, data=json.dumps(params), headers=headers, timeout=100)
    resp = json.loads(create_resp.text)
    print("response is: ", resp)
    result = resp['result']
    #result = json.loads(create_resp.text)['result']
    return result

def request_parimary_key_ciphertext(ip, port):
    action = "CreateKey"
    payload = {
        "keyspec":"EH_AES_GCM_128",
        "origin":"EH_INTERNAL_KEY"
    }
    result = post_request(ip, port, action, payload)
    print("result is: ", result)
    primary_key_ciphertext = result['keyid']

    #primary_key_ciphertext = post_request(ip, port, action, payload)['cmk_base64']
    return primary_key_ciphertext

def request_data_key_ciphertext(ip, port, encrypted_primary_key):
    action = "GenerateDataKeyWithoutPlaintext"
    payload = {
        "keyid":encrypted_primary_key,
        "keylen": 32,
        "aad": "test",
    }
    result = post_request(ip, port, action, payload)
    data_key_ciphertext = result['ciphertext']
    return data_key_ciphertext

def request_data_key_plaintext(ip, port, encrypted_primary_key, encrypted_data_key):
    action = "Decrypt"
    payload = {
        "keyid":encrypted_primary_key,
        "ciphertext":encrypted_data_key,
        "aad":"test",
    }
    data_key_plaintext = post_request(ip, port, action, payload)['plaintext']
    return data_key_plaintext
