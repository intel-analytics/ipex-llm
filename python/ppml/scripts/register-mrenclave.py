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
import argparse
import base64
import time
import random
import hmac
from hashlib import sha256
from collections import OrderedDict
import urllib.parse

use_secure_cert = False
headers = {"Content-Type":"application/json"}
# below placeholders for global variables, and they will be initalized in the below context
appid = ''
apikey = ''

def init_appid_apikey(appid_user, apikey_user):
    global appid
    global apikey
    appid = appid_user
    apikey = apikey_user

def init_params(payload=False):
    params = OrderedDict()
    params["appid"] = appid
    if payload!=False:
        ord_payload = OrderedDict(sorted(payload.items(), key=lambda k: k[0]))
        params["payload"] = urllib.parse.unquote_plus(urllib.parse.urlencode(ord_payload))
    params["timestamp"] = str(int(time.time() * 1000))
    sign_string = urllib.parse.unquote_plus(urllib.parse.urlencode(params))
    sign = str(base64.b64encode(hmac.new(apikey.encode('utf-8'), sign_string.encode('utf-8'), digestmod=sha256).digest()),'utf-8')
    if payload!=False:
        params["payload"] = payload
    params["sign"] = sign
    return params

def no_bool_convert(pairs):
  return {k: str(v).casefold() if isinstance(v, bool) else v for k, v in pairs}

def check_result(res, action):
    res_json = json.loads(res.text)
    if(res_json['code'] == 200):
        print("%s successfully \n" %(action))
        return True
    else:
        print("%s failed, error message: %s \n" %(action, res_json["message"]))
        return False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--appid', type=str, help='your ehsm appid obtained from enroll', required=True)
    parser.add_argument('--apikey', type=str, help='your ehsm apikey obtained from enroll', required=True)
    parser.add_argument('--url', type=str, help='the address of the ehsm_kms_server, fornmat likes https://1.2.3.4', required=True)
    parser.add_argument('--mr_enclave', type=str, help='mr_enclave you will upload', required=True)
    parser.add_argument('--mr_signer', type=str, help='mr_signer you will upload', required=True)
    args = parser.parse_args()

    base_url = args.url + "/ehsm?Action="
    print(base_url)
    return base_url, args.mr_enclave, args.mr_signer, args.appid, args.apikey

def register(base_url, mr_enclave, mr_signer, appid, apikey):
    payload = OrderedDict()
    payload["mr_enclave"] = mr_enclave
    payload["mr_signer"] = mr_signer
    init_appid_apikey(appid, apikey)
    params = init_params(payload)
    print('register req:\n%s\n' %(params))

    resp = requests.post(url=base_url + "UploadQuotePolicy", data=json.dumps(params), headers=headers, verify=use_secure_cert)
    if(check_result(resp, 'UploadQuotePolicy') == False):
        return
    print('register resp:\n%s\n' %(resp.text))

    policyId = json.loads(resp.text)['result']['policyId']
    return policyId

if __name__ == "__main__":

    base_url, mr_enclave, mr_signer, appid, apikey = get_args()

    register(base_url, mr_enclave, mr_signer, appid, apikey)
