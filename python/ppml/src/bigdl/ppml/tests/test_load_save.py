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

import torch.nn.functional as F
import torch, os, io
import torch.nn as nn
import pytest
from ptorch.models import save, load
from kms.client import decrypt_buf_with_key, generate_primary_key, generate_data_key

class linearModel(nn.Module):
    def __init__(self):
        super(linearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        # Let's fill in the weight by ourselves so that we can later change it.
        self.linear.weight.data.fill_(1.245)
    
    def forward(self, x):
        out = self.linear(x)
        return out


os.environ['APPID'] = "63a88858-29f6-426f-b9b7-15702bf056ac"
os.environ['APIKEY'] = "PxiX0hduXAG76cw1JMYPJWyGBMGc0muB"


encrypted_primary_key_path = ""
encrypted_data_key_path = ""

ehsm_ip = "172.168.0.226"
ehsm_port = "9000"


@pytest.fixture(scope="session", autouse=True)
def prepare_test_env():
    # Prepare the keys
    generate_primary_key(ehsm_ip, ehsm_port)
    global encrypted_primary_key_path
    encrypted_primary_key_path = "./encrypted_primary_key"
    generate_data_key(ehsm_ip, ehsm_port, encrypted_primary_key_path, 32)
    global encrypted_data_key_path
    encrypted_data_key_path = "./encrypted_data_key"

def test_save_file_method():
    model = linearModel()
    encrypted_buf = io.BytesIO()
    # Reference expected value
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

def test_save_load():
    # Initialize the model
    model = linearModel()
    save(model.state_dict(), "testsave.pt", ehsm_ip, ehsm_port, encrypted_primary_key_path, encrypted_data_key_path)
    model.linear.weight.data.fill_(1.110)
    # now we try to load it back, and check the weight is the same
    model.load_state_dict(load("testsave.pt", ehsm_ip, ehsm_port, encrypted_primary_key_path, encrypted_data_key_path))
    assert model.linear.weight.data[0] == 1.245

def test_save_load_buf():
    # Initialize the model
    model = linearModel()
    buf = io.BytesIO()
    save(model.state_dict(), buf, ehsm_ip, ehsm_port, encrypted_primary_key_path, encrypted_data_key_path)
    model.linear.weight.data.fill_(1.110)
    # now we try to load it back, and check the weight is the same
    model.load_state_dict(load(buf, ehsm_ip, ehsm_port, encrypted_primary_key_path, encrypted_data_key_path))
    assert model.linear.weight.data[0] == 1.245


import torch.optim as optim
# The example from pytorch tutorial
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def test_multi_save():
    model = TheModelClass()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    save({
        'epoch': 5,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': 1.842,
    }, "checkpoint.pt", ehsm_ip, ehsm_port, encrypted_primary_key_path, encrypted_data_key_path)
    checkpoint = load("checkpoint.pt", ehsm_ip, ehsm_port, encrypted_primary_key_path, encrypted_data_key_path)
    assert checkpoint['epoch'] == 5
    assert checkpoint['loss'] == 1.842
    assert optimizer.state_dict() == checkpoint['optimizer_state_dict']
    for param_tensor in model.state_dict():
        assert model.state_dict()[param_tensor] == checkpoint['optimizer_state_dict'][param_tensor]