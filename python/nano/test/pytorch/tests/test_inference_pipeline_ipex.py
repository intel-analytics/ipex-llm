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

import os
from torch import nn
import torch
from unittest import TestCase
import pytest
import torchvision.transforms as transforms
from bigdl.nano.pytorch import Trainer
from bigdl.nano.pytorch import InferenceOptimizer
import torchmetrics
import torch
import torch.nn.functional as F
from test.pytorch.utils._train_torch_lightning import create_data_loader


data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


class Net(nn.Module):
    def __init__(self, l1=8, l2=16):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TestInferencePipeline(TestCase):
    num_workers = 0
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    metric = torchmetrics.Accuracy(num_classes=10, top_k=1)
    max_epochs = 5

    model = Net()
    test_loader = create_data_loader(data_dir, 1, num_workers, data_transform, subset=10, shuffle=False)
    train_loader = create_data_loader(data_dir, 32, num_workers, data_transform, subset=10, shuffle=True)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(max_epochs=max_epochs)
    model = Trainer.compile(model, loss, optimizer)
    trainer.fit(model, train_loader)
    
    def test_get_model_without_optimize(self):
        inference_opt = InferenceOptimizer()
        with pytest.raises(RuntimeError) as e:
            acc_model, option = inference_opt.get_best_model()
        error_msg = e.value.args[0]
        assert error_msg == "There is no optimized model. You should call .optimize() " \
                            "before get_best_model()"

    def test_pipeline_with_metric(self):
        inference_opt = InferenceOptimizer()
        inference_opt.optimize(model=self.model,
                               training_data=self.train_loader,
                               validation_data=self.test_loader,
                               metric=self.metric,
                               direction="max",
                               thread_num=1)

        acc_model, option = inference_opt.get_best_model()
        acc_model, option = inference_opt.get_best_model(accelerator="onnxruntime")
        assert option == "" or "onnxruntime" in option
        acc_model, option = inference_opt.get_best_model(precision="int8")
        assert option == "" or "inc" in option or "pot" in option
        acc_model, option = inference_opt.get_best_model(accuracy_criterion=0.1)
        acc_model(next(iter(self.train_loader))[0])

    def test_pipeline_without_metric(self):
        inference_opt = InferenceOptimizer()
        inference_opt.optimize(model=self.model,
                               training_data=self.train_loader,
                               thread_num=1)

        acc_model, option = inference_opt.get_best_model()
        acc_model, option = inference_opt.get_best_model(accelerator="onnxruntime")
        assert option == "" or "onnxruntime" in option
        acc_model, option = inference_opt.get_best_model(precision="int8")
        assert option == "" or "inc" in option or "pot" in option
        with pytest.raises(RuntimeError) as e:
            acc_model, option = inference_opt.get_best_model(accuracy_criterion=0.1)
        error_msg = e.value.args[0]
        assert error_msg == "If you want to specify accuracy_criterion, you need "\
                            "to set metric and validation_data when call 'optimize'."

    def test_summary(self):
        inference_opt = InferenceOptimizer()
        with pytest.raises(RuntimeError) as e:
            inference_opt.summary()
        error_msg = e.value.args[0]
        assert error_msg == "There is no optimization result. You should call .optimize() "\
                            "before summary()"
        inference_opt.optimize(model=self.model,
                               training_data=self.train_loader,
                               thread_num=1)
        inference_opt.summary()

    def test_wrong_data_loader(self):
        fake_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize(64),
        ])
        fake_train_loader = create_data_loader(self.data_dir, 32, self.num_workers, 
                                               fake_transform, subset=10, shuffle=True)
        inference_opt = InferenceOptimizer()
        with pytest.raises(RuntimeError) as e:
            inference_opt.optimize(model=self.model,
                                    training_data=fake_train_loader,
                                    thread_num=1)
        error_msg = e.value.args[0]
        assert error_msg == "training_data is incompatible with your model input."
