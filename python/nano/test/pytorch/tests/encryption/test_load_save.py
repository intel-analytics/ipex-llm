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
import unittest
import random
from bigdl.nano.pytorch.patching import patch_encryption
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

class linearModel(nn.Module):
    def __init__(self):
        super(linearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        # Let's fill in the weight by ourselves so that we can later change it.
        self.linear.weight.data.fill_(1.245)
    
    def forward(self, x):
        out = self.linear(x)
        return out

def _create_random(length) -> str:
    ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chars = []
    for i in range(length):
        chars.append(random.choice(ALPHABET))
    key = "".join(chars)
    key_bytes = key.encode("ascii")
    base64_str = base64.b64encode(key_bytes)
    print(len(base64_str), flush=True)
    return base64_str

encryption_key = _create_random(32)

class TestModelSaveLoad(unittest.TestCase):

    def setUp(self) -> None:
        patch_encryption()

    def test_save_load_to_buf(self):
        model = linearModel()
        encrypted_buf = io.BytesIO()
        # Reference expected value
        expected_buf = io.BytesIO()
        torch.old_save(model.state_dict(), expected_buf)
        expected_buf.seek(0)
        expected_state = torch.load(expected_buf)
        torch.save(model.state_dict(), encrypted_buf, encryption_key=encryption_key)
        # load it back and compare the stat_dict is the same
        our_state_dict = torch.load(encrypted_buf, decryption_key=encryption_key)
        self.assertEqual(our_state_dict, expected_state)

    def test_save_load_to_file(self):
        # Initialize the model
        model = linearModel()
        torch.save(model.state_dict(), "testsave.pt", encryption_key=encryption_key)
        # change its default value
        model.linear.weight.data.fill_(1.110)
        # now we try to load it back, and check the weight is the same
        model.load_state_dict(torch.load("testsave.pt", decryption_key=encryption_key))
        self.assertEqual(model.linear.weight.data[0], 1.245)

    def test_save_load_buf2(self):
        # Initialize the model
        model = linearModel()
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf, encryption_key=encryption_key)
        model.linear.weight.data.fill_(1.110)
        # now we try to load it back, and check the weight is the same
        model.load_state_dict(torch.load(buf, decryption_key=encryption_key))
        self.assertEqual(model.linear.weight.data[0], 1.245)

    def test_multi_save(self):
        model = TheModelClass()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        torch.save({
            'epoch': 5,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': 1.842,
        }, "checkpoint.pt", encryption_key=encryption_key)
        checkpoint = torch.load("checkpoint.pt", decryption_key=encryption_key)
        self.assertEqual(checkpoint['epoch'], 5)
        self.assertEqual(checkpoint['loss'], 1.842)
        self.assertEqual(optimizer.state_dict(), checkpoint['optimizer_state_dict'])
        for param_tensor in model.state_dict():
            self.assertTrue(torch.equal(model.state_dict()[param_tensor], checkpoint['model_state_dict'][param_tensor]))

    def test_without_keys(self):
        model = TheModelClass()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        torch.save({
            'epoch': 5,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': 1.842,
        }, "checkpoint.pt")
        checkpoint = torch.load("checkpoint.pt")
        self.assertEqual(checkpoint['epoch'], 5)
        self.assertEqual(checkpoint['loss'], 1.842)
        self.assertEqual(optimizer.state_dict(), checkpoint['optimizer_state_dict'])
        for param_tensor in model.state_dict():
            self.assertTrue(torch.equal(model.state_dict()[
                            param_tensor], checkpoint['model_state_dict'][param_tensor]))
