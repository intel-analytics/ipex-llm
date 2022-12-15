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

import torch, os, io
from ptorch.models import save, load
from kms.client import decrypt_buf_with_key

class linearModel(torch.nn.Module):
    def __init__(self):
        super(linearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
        # Let's fill in the weight by ourselves so that we can later change it.
        self.linear.weight.data.fill_(1.245)
    
    def forward(self, x):
        out = self.linear(x)
        return out


os.environ['APPID'] = "63a88858-29f6-426f-b9b7-15702bf056ac"
os.environ['APIKEY'] = "PxiX0hduXAG76cw1JMYPJWyGBMGc0muB"
encrypted_primary_key_path = "/ppml/encrypted_primary_key"
encrypted_data_key_path = "/ppml/encrypted_data_key"
ehsm_ip = "172.168.0.226"
ehsm_port = "9000"

def test_save_method():
    model = linearModel()
    encrypted_buf = io.BytesIO()
    # Create the expected buffer
    expected_buf = io.BytesIO()
    torch.save(model.state_dict(), expected_buf)

    save(model.state_dict(), encrypted_buf, ehsm_ip, ehsm_port, encrypted_primary_key_path, encrypted_data_key_path)
    our_buf = io.BytesIO()
    decrypt_buf_with_key(encrypted_buf, our_buf, ehsm_ip, ehsm_port, encrypted_primary_key_path, encrypted_data_key_path)
    assert our_buf.getvalue() == expected_buf.getvalue()
    # Test write it to a file and load it back, should get same value
    with open("testmodel.pt", 'wb') as opened_file:
        opened_file.write(our_buf.getvalue())
        opened_file.flush()
        opened_file.close()

    read_buf = io.BytesIO()
    with open("testmodel.pt", 'rb') as opened_file:
        read_buf.write(opened_file.read())
    assert read_buf.getvalue() == expected_buf.getvalue()

    with open("testmodel.pt", 'wb') as opened_file:
        opened_file.write(encrypted_buf.getvalue())
        opened_file.flush()
        opened_file.close()

    read_buf = io.BytesIO()
    with open("testmodel.pt", 'rb') as opened_file:
        read_buf.write(opened_file.read())
    our_buf = io.BytesIO() 
    decrypt_buf_with_key(read_buf, our_buf, ehsm_ip, ehsm_port, encrypted_primary_key_path, encrypted_data_key_path)
    assert our_buf.getvalue() == expected_buf.getvalue()


def original_save_load():
    model = linearModel()
    model.linear.weight.data.fill_(1.110)
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    # now we try to load it back, and check the weight is the same
    model.load_state_dict(torch.load(buf))
    buf.close()
    print(model.linear.weight.data)

def save_load():
    # Initialize the model
    model = linearModel()
    save(model.state_dict(), "testsave.pt", ehsm_ip, ehsm_port, encrypted_primary_key_path, encrypted_data_key_path)
    # now we try to load it back, and check the weight is the same
    model.load_state_dict(load("testsave.pt", ehsm_ip, ehsm_port, encrypted_primary_key_path, encrypted_data_key_path))
    print(model.linear.weight.data)
