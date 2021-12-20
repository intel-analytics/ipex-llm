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
    args = parser.parse_args()
    ip = args.ip
    port = args.port
    return ip, port

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

def generate_primary_key(ip, port):
    base_url = "http://" + ip + ":" + port + "/ehsm?Action="
    headers = {"Content-Type":"application/json"}
    params=request_params({
            "keyspec":"EH_AES_GCM_128",
            "origin":"EH_INTERNAL_KEY"
        })
    create_resp = requests.post(url=base_url + "CreateKey", data=json.dumps(params), headers=headers, timeout=100)
    print('Writing generated primary key to local file...')
    with open("./encrypted_primary_key", "w") as f:
        f.write(json.loads(create_resp.text)['result']['cmk_base64'])
    print('Generate Primary Key Successfully!')

if __name__ == "__main__":
    ip,port = get_args()
    generate_primary_key(ip, port)
