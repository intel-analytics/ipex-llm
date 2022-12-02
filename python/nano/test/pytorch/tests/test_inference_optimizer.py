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
import numpy as np
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
from bigdl.nano.utils.log4Error import invalidOperationError


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
        assert option == "original" or "onnxruntime" in option
        acc_model, option = inference_opt.get_best_model(precision="int8")
        assert option == "original" or "inc" in option or "int8" in option
        acc_model, option = inference_opt.get_best_model(accuracy_criterion=0.1)
        acc_model(next(iter(self.train_loader))[0])

    def test_pipeline_without_metric(self):
        inference_opt = InferenceOptimizer()
        inference_opt.optimize(model=self.model,
                               training_data=self.train_loader,
                               thread_num=1)

        acc_model, option = inference_opt.get_best_model()
        acc_model, option = inference_opt.get_best_model(accelerator="onnxruntime")
        assert option == "original" or "onnxruntime" in option
        acc_model, option = inference_opt.get_best_model(precision="int8")
        assert option == "original" or "inc" in option or "int8" in option
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
                               excludes=["bf16", "original"])

        # original is a special method that must be included in
        # the search
        assert "original" in inference_opt.optimized_model_dict
        assert "jit_fp32_ipex" in inference_opt.optimized_model_dict
        assert "bf16" not in inference_opt.optimized_model_dict

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
        if TORCH_VERSION_LESS_1_10:
            return
        # test automatic add label for quantization
        optim_dict = inference_opt.optimized_model_dict
        assert optim_dict["openvino_int8"]["status"] in ("successful", "early_stopped")
        assert optim_dict["onnxruntime_int8_qlinear"]["status"] in ("successful", "early_stopped")

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
        for model_class in [MultipleInputNet, MultipleInputWithKwargsNet]:
            net = model_class()
            x1 = torch.randn(32, 10)
            x2 = torch.randn(32, 10)
            y = torch.randn(32, 1)
            if isinstance(net, MultipleInputNet):
                dataloader = DataLoader(TensorDataset(x1, x2, y), batch_size=1)
            else:
                x3 = torch.randn(32, 1)
                dataloader = DataLoader(TensorDataset(x1, x2, x3, y), batch_size=1)

            # int8
            InferenceOptimizer.quantize(net,
                                        calib_data=dataloader)

            # int8-onnxruntime
            InferenceOptimizer.quantize(net,
                                        accelerator="onnxruntime",
                                        calib_data=dataloader)

            # int8-onnxruntime
            InferenceOptimizer.trace(net,
                                     accelerator="onnxruntime",
                                     input_sample=dataloader)

            # openvino
            InferenceOptimizer.trace(net,
                                     accelerator="openvino",
                                     input_sample=dataloader)

    def test_pipeline_with_tensor_accuracy(self):
        inference_opt = InferenceOptimizer()

        def metric(model, data_loader):
            metrics = []
            for input_data, target in data_loader:
                pred = model(input_data)
                metric = self.metric(pred, target)
                metrics.append(metric)
            return torch.FloatTensor([np.mean(metrics)])

        inference_opt.optimize(model=self.model,
                               training_data=self.train_loader,
                               validation_data=self.test_loader,
                               metric=metric,
                               direction="max",
                               thread_num=4)

    def test_multi_instance(self):
        model = Net()
        model.eval()
        inference_opt = InferenceOptimizer()
        multi_instance_model = inference_opt.to_multi_instance(model, num_processes=4)

        test_loader = create_data_loader(self.data_dir, 1, self.num_workers, data_transform,
                                         subset=50, shuffle=False)
        input_data = list(map(lambda b: b[0], test_loader))

        with torch.no_grad():
            preds1 = multi_instance_model(input_data)
            preds2 = [model(b) for b in input_data]

        for (pred1, pred2) in zip(preds1, preds2):
            np.testing.assert_allclose(pred1, pred2, atol=1e-4,
                                        err_msg=f"\npred1: {pred1}\npred2: {pred2}\n")

    def test_grid_search_model_with_accelerator(self):
        inference_opt = InferenceOptimizer()

        inference_opt.optimize(model=self.model,
                               training_data=self.train_loader,
                               validation_data=self.test_loader,
                               metric=self.metric,
                               direction="max",
                               thread_num=4,
                               accelerator=("openvino", ))
        optim_dict = inference_opt.optimized_model_dict
        assert len(optim_dict) == 3
        with pytest.raises(RuntimeError):
            acc_model, option = inference_opt.get_best_model(accelerator="onnxruntime")

    def test_grid_search_model_with_precision(self):
        inference_opt = InferenceOptimizer()

        inference_opt.optimize(model=self.model,
                               training_data=self.train_loader,
                               validation_data=self.test_loader,
                               metric=self.metric,
                               direction="max",
                               thread_num=4,
                               precision=('bf16', 'int8'))
        optim_dict = inference_opt.optimized_model_dict
        assert len(optim_dict) == 14
        with pytest.raises(RuntimeError):
            acc_model, option = inference_opt.get_best_model(precision="fp32")

    def test_default_search_mode(self):
        inference_opt = InferenceOptimizer()

        inference_opt.optimize(model=self.model,
                               training_data=self.train_loader,
                               validation_data=self.test_loader,
                               metric=self.metric,
                               direction="max",
                               thread_num=4,
                               search_mode="default")
        optim_dict = inference_opt.optimized_model_dict
        assert len(optim_dict) == 11

    def test_pipeline_with_single_tensor_loader(self):
        input_sample = torch.rand(10, 3, 32, 32)
        dataloader = DataLoader(input_sample, batch_size=1)
        inference_opt = InferenceOptimizer()
        inference_opt.optimize(model=self.model,
                               training_data=dataloader,
                               thread_num=1,
                               latency_sample_num=10)
        if TORCH_VERSION_LESS_1_10:
            return
        # test automatic add label for quantization
        optim_dict = inference_opt.optimized_model_dict
        assert optim_dict["openvino_int8"]["status"] in ("successful", "early_stopped")
        assert optim_dict["onnxruntime_int8_qlinear"]["status"] in ("successful", "early_stopped")

    def test_context_manager(self):
        inference_opt = InferenceOptimizer()
        inference_opt.optimize(model=self.model,
                               training_data=self.train_loader,
                               validation_data=self.test_loader,
                               metric=self.metric,
                               direction="max",
                               search_mode="all")
        # test builtin optimized_model_dict
        optim_dict = inference_opt.optimized_model_dict
        for method, option in optim_dict.items():
            if option["status"] == "successful":
                model = option["model"]
                with InferenceOptimizer.get_context(model):
                    pass
        # test get_model
        for method in list(InferenceOptimizer.ALL_INFERENCE_ACCELERATION_METHOD.keys()):
            if "model" in optim_dict[method]:
                model = inference_opt.get_model(method)
                with InferenceOptimizer.get_context(model):
                    pass
        # test get_best_model
        model, option = inference_opt.get_best_model()
        with InferenceOptimizer.get_context(model):
            pass

    def test_inplace(self):
        class CannotCopyNet(Net):
            def __deepcopy__(self, memo):
                invalidOperationError(False, "The `deepcopy` function shouldn't be called")

        inference_opt = InferenceOptimizer()
        # bf16+ipex
        model = CannotCopyNet()
        bf16_ipex_model = inference_opt.quantize(model, calib_data=self.train_loader, precision='bf16', use_ipex=True, inplace=True)
        # ipex
        model = CannotCopyNet()
        ipex_model = inference_opt.trace(model, input_sample=self.train_loader, use_ipex=True, inplace=True)

        inference_opt.save(bf16_ipex_model, "bf16_ipex")
        inference_opt.save(ipex_model, "ipex")

        bf16_ipex_model = inference_opt.load("bf16_ipex", model, inplace=True)
        ipex_model = inference_opt.load("ipex", model, inplace=True)
