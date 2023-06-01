import requests
import json
import base64
import generate_tdx_quote
from collections import OrderedDict
use_secure_cert = False
headers = {"Content-Type":"application/json"}

def attest_with_bigdl_as(base_url, app_id, api_key, quote, policy_id):
    payload = OrderedDict()
    payload["appID"] = app_id
    payload["apiKey"] = api_key
    payload["quote"] = base64.b64encode(quote).decode()
    if len(policy_id) > 0:
        payload["policyID"] = policy_id
    resp = requests.post(url=base_url + "/verifyQuote", data=json.dumps(payload), headers=headers, verify=use_secure_cert)
    print(json.loads(resp.text)["result"])
    return json.loads(resp.text)["result"]

