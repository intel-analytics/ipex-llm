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
from requests.packages import urllib3

# default app auth conf
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
appid = os.getenv('APPID')
apikey = os.getenv('APIKEY')
headers = {"Content-Type":"application/json"}
use_secure_cert = False

def request_params(payload):
    params = OrderedDict()
    params["appid"] = appid
    ord_payload = OrderedDict(sorted(payload.items(), key=lambda k: k[0]))
    params["payload"] = urllib.parse.unquote(urllib.parse.urlencode(ord_payload))
    params["timestamp"] = str(int(time.time()*1000))
    sign_string = urllib.parse.unquote(urllib.parse.urlencode(params))
    sign = str(base64.b64encode(hmac.new(apikey.encode('utf-8'),
                                         sign_string.encode('utf-8'),
                                         digestmod=sha256).digest()),'utf-8')
    params["payload"] = ord_payload
    params["sign"] = sign
    return params

def construct_url(ip, port, action):
    return "https://" + ip + ":" + port + "/ehsm?Action=" + action

def post_request(ip, port, action, payload):
    url = construct_url(ip, port, action)
    params = request_params(payload)
    create_resp = requests.post(url=url, data=json.dumps(params), headers=headers, timeout=100, verify=use_secure_cert)
    result = json.loads(create_resp.text)['result']
    return result

def request_parimary_key_ciphertext(ip, port):
    action = "CreateKey"
    payload = {
        "keyspec":"EH_AES_GCM_128",
        "origin":"EH_INTERNAL_KEY"
    }
    primary_key_ciphertext = post_request(ip, port, action, payload)['keyid']
    return primary_key_ciphertext

def request_data_key_ciphertext(ip, port, encrypted_primary_key, data_key_length):
    action = "GenerateDataKeyWithoutPlaintext"
    payload = {
        "keyid":encrypted_primary_key,
        "keylen": data_key_length,
        "aad": "test",
    }
    data_key_ciphertext = post_request(ip, port, action, payload)['ciphertext']
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
