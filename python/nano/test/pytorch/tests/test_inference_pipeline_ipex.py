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
from torch.utils.data import TensorDataset, DataLoader
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10


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


class MultipleInputNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(10, 1)
        self.dense2 = nn.Linear(10, 1)

    def forward(self, x1, x2):
        return self.dense1(x1) + self.dense2(x2)


class MultipleInputWithKwargsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(10, 1)
        self.dense2 = nn.Linear(10, 1)

    def forward(self, x1, x2, x3=10):
        return self.dense1(x1) + self.dense2(x2) + x3


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
        assert option == "" or "inc" in option or "int8" in option
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
        assert option == "" or "inc" in option or "int8" in option
        with pytest.raises(RuntimeError) as e:
            acc_model, option = inference_opt.get_best_model(accuracy_criterion=0.1)
        error_msg = e.value.args[0]
        assert error_msg == "If you want to specify accuracy_criterion, you need "\
                            "to set metric and validation_data when call 'optimize'."

    def test_pipeline_with_excludes(self):
        inference_opt = InferenceOptimizer()
        inference_opt.optimize(model=self.model,
                               training_data=self.train_loader,
                               thread_num=1,
                               excludes=["fp32_ipex", "original"])

        # original is a special method that must be included in
        # the search
        assert "original" in inference_opt.optimized_model_dict
        assert "jit_fp32_ipex" in inference_opt.optimized_model_dict
        assert "fp32_ipex" not in inference_opt.optimized_model_dict

    def test_pipeline_with_includes(self):
        inference_opt = InferenceOptimizer()
        inference_opt.optimize(model=self.model,
                               training_data=self.train_loader,
                               thread_num=1,
                               includes=["fp32_ipex"])

        assert "original" in inference_opt.optimized_model_dict
        assert "fp32_ipex" in inference_opt.optimized_model_dict
        assert len(inference_opt.optimized_model_dict) == 2

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

    def test_pipeline_with_custom_function_metric(self):
        inference_opt = InferenceOptimizer()

        def metric(pred, target):
            return self.metric(pred, target)

        inference_opt.optimize(model=self.model,
                               training_data=self.train_loader,
                               validation_data=self.test_loader,
                               metric=metric,
                               direction="max",
                               thread_num=1)
    
    def test_pipeline_with_torchmetrics_functional_metric(self):
        inference_opt = InferenceOptimizer()
        metric = torchmetrics.functional.accuracy
        inference_opt.optimize(model=self.model,
                               training_data=self.train_loader,
                               validation_data=self.test_loader,
                               metric=metric,
                               direction="max",
                               thread_num=1)

    def test_pipeline_with_custom_function_metric_without_data(self):
        inference_opt = InferenceOptimizer()

        def metric(pred, target):
            return self.metric(pred, target)

        with pytest.raises(RuntimeError):
            inference_opt.optimize(model=self.model,
                                training_data=self.train_loader,
                                validation_data=None,
                                metric=metric,
                                direction="max",
                                thread_num=1)

    def test_pipeline_with_wrong_custom_function_metric(self):
        inference_opt = InferenceOptimizer()

        def metric(x, y):
            return self.metric(x, y)

        with pytest.raises(RuntimeError):
            inference_opt.optimize(model=self.model,
                                training_data=self.train_loader,
                                validation_data=self.test_loader,
                                metric=metric,
                                direction="max",
                                thread_num=1)

    def test_pipeline_with_custom_function_metric_with_data_loader(self):
        inference_opt = InferenceOptimizer()
        import numpy as np
        def metric(model, data_loader):
            metrics = []
            for input_data, target in data_loader:
                pred = model(input_data)
                metric = self.metric(pred, target)
                metrics.append(metric)
            return np.mean(metrics)

        inference_opt.optimize(model=self.model,
                               training_data=self.train_loader,
                               validation_data=self.test_loader,
                               metric=metric,
                               direction="max",
                               thread_num=1)

    def test_get_model_with_wrong_method_name(self):
        inference_opt = InferenceOptimizer()
        inference_opt.optimize(model=self.model,
                               training_data=self.train_loader,
                               validation_data=self.test_loader,
                               metric=self.metric,
                               direction="max",
                               thread_num=1)

        with pytest.raises(RuntimeError):
            inference_opt.get_model(method_name="fp16_ipex")

    def test_get_model_with_method_name(self):
        inference_opt = InferenceOptimizer()
        inference_opt.optimize(model=self.model,
                               training_data=self.train_loader,
                               validation_data=self.test_loader,
                               metric=self.metric,
                               direction="max",
                               thread_num=1)
        try:
            model = inference_opt.get_model(method_name="fp32_ipex")
            from bigdl.nano.deps.ipex.ipex_inference_model import PytorchIPEXJITModel
            assert isinstance(model, PytorchIPEXJITModel)
        except:
            pass

    def test_pipeline_with_single_tensor(self):
        input_sample = torch.rand(1, 3, 32, 32)
        inference_opt = InferenceOptimizer()
        inference_opt.optimize(model=self.model,
                               training_data=input_sample,
                               thread_num=1,
                               latency_sample_num=10)

    def test_pipeline_with_single_tuple_of_tensor(self):
        input_sample = (torch.rand(1, 3, 32, 32), torch.Tensor([1]).int())
        inference_opt = InferenceOptimizer()
        inference_opt.optimize(model=self.model,
                               training_data=input_sample,
                               thread_num=1,
                               latency_sample_num=10)

    def test_pipeline_accuracy_with_single_tuple_of_tensor(self):
        input_sample = (torch.rand(1, 3, 32, 32), torch.Tensor([1]).int())
        inference_opt = InferenceOptimizer()
        inference_opt.optimize(model=self.model,
                               training_data=input_sample,
                               validation_data=input_sample,
                               metric=self.metric,
                               thread_num=1,
                               latency_sample_num=10)

    def test_multiple_input_dataloader(self):
        # will not run this test if torch < 1.10
        if TORCH_VERSION_LESS_1_10:
            return

        for model_class in [MultipleInputNet, MultipleInputWithKwargsNet]:
            net = model_class()
            x1 = torch.randn(32, 10)
            x2 = torch.randn(32, 10)
            y = torch.randn(32, 1)
            dataloader = DataLoader(TensorDataset(x1, x2, y), batch_size=1)

            # int8
            InferenceOptimizer.quantize(net,
                                        calib_dataloader=dataloader)

            # int8-onnxruntime
            InferenceOptimizer.quantize(net,
                                        accelerator="onnxruntime",
                                        calib_dataloader=dataloader)

            # int8-onnxruntime
            InferenceOptimizer.trace(net,
                                    accelerator="onnxruntime",
                                    input_sample=dataloader)

            # int8-openvino
            InferenceOptimizer.trace(net,
                                    accelerator="openvino",
                                    input_sample=dataloader)
