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

import torch.optim as optim
import torch.nn.functional as F
import os
import torch.nn as nn
from bigdl.ppml.encryption.torch.models import save, load
from bigdl.ppml.kms.client import generate_primary_key, generate_data_key, get_data_key_plaintext


class linearModel(nn.Module):
    def __init__(self):
        super(linearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        # Let's fill in the weight by ourselves so that we can later change it.
        self.linear.weight.data.fill_(1.245)

    def forward(self, x):
        out = self.linear(x)
        return out


# Define APPID and APIKEY in os.environment
APPID = os.environ.get('APPID')
APIKEY = os.environ.get('APIKEY')


encrypted_primary_key_path = ""
encrypted_data_key_path = ""

EHSM_IP = os.environ.get('ehsm_ip')
EHSM_PORT = os.environ.get('ehsm_port', "9000")

# prepare environment
def prepare_env():
    if APPID is None or APIKEY is None or EHSM_IP is None:
        print("Please set environment variable APPID, APIKEY, ehsm_ip!")
        exit(1)
    generate_primary_key(EHSM_IP, EHSM_PORT)
    global encrypted_primary_key_path
    encrypted_primary_key_path = "./encrypted_primary_key"
    generate_data_key(EHSM_IP, EHSM_PORT, encrypted_primary_key_path, 32)
    global encrypted_data_key_path
    encrypted_data_key_path = "./encrypted_data_key"

# Get a key from kms that can be used for encryption/decryption
def get_key():
    return get_data_key_plaintext(EHSM_IP, EHSM_PORT, encrypted_primary_key_path, encrypted_data_key_path)

def main():
    prepare_env()
    # This is only safe in sgx environment, where the memory can not be read
    secret_key = get_key()
    # Try to save a model
    model = linearModel()
    save(model.state_dict(), "testsave.pt", secret_key)
    model.linear.weight.data.fill_(1.110)
    model.load_state_dict(load("testsave.pt", secret_key))
    # Should print 1.245
    print(model.linear.weight.data[0], flush=True)

if __name__ == "__main__":
    main()
