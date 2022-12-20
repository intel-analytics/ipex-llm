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
import base64
import torch.nn as nn
import pytest
import random
from bigdl.ppml.encryption.torch.models import save, load
from bigdl.ppml.kms.client import decrypt_buffer_with_key, generate_primary_key, generate_data_key

def _create_random(length) -> str:
    ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chars = []
    for i in range(length):
        chars.append(random.choice(ALPHABET))
    key = "".join(chars)
    base64_str = base64.b64encode(key)
    print(len(base64_str), flush=True)
    return base64_str



class linearModel(nn.Module):
    def __init__(self):
        super(linearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        # Let's fill in the weight by ourselves so that we can later change it.
        self.linear.weight.data.fill_(1.245)
    
    def forward(self, x):
        out = self.linear(x)
        return out

encryption_key = ""

@pytest.fixture(scope="session", autouse=True)
def prepare_test_env():
    global encryption_key
    encryption_key = _create_random(32)

def test_save_load_to_buf():
    model = linearModel()
    encrypted_buf = io.BytesIO()
    # Reference expected value
    expected_buf = io.BytesIO()
    torch.save(model.state_dict(), expected_buf)
    expected_state = torch.load(expected_buf)

    save(model.state_dict(), encrypted_buf, encryption_key)
    # load it back and compare the stat_dict is the same
    our_state_dict = load(encrypted_buf, encryption_key)

    assert our_state_dict == expected_state

def test_save_load_to_file():
    # Initialize the model
    model = linearModel()
    save(model.state_dict(), "testsave.pt", encryption_key)
    # change its default value
    model.linear.weight.data.fill_(1.110)
    # now we try to load it back, and check the weight is the same
    model.load_state_dict(load("testsave.pt", encryption_key))
    assert model.linear.weight.data[0] == 1.245

def test_save_load_buf2():
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
    }, "checkpoint.pt", encryption_key)
    checkpoint = load("checkpoint.pt", encryption_key)
    assert checkpoint['epoch'] == 5
    assert checkpoint['loss'] == 1.842
    assert optimizer.state_dict() == checkpoint['optimizer_state_dict']
    for param_tensor in model.state_dict():
        assert torch.equal(model.state_dict()[param_tensor], checkpoint['model_state_dict'][param_tensor])
