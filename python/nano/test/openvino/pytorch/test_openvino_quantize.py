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
from unittest import TestCase
from torchmetrics import F1Score
from bigdl.nano.pytorch import Trainer
from bigdl.nano.pytorch import InferenceOptimizer
from torchvision.models.mobilenetv3 import mobilenet_v3_small
from torchvision.models import resnet50
import torch
from torch import nn
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
import tempfile
import pytest
import numpy as np


class TestOpenVINO(TestCase):
    def test_quantize_openvino(self):
        trainer = Trainer()
        model = mobilenet_v3_small(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)

        ds = TensorDataset(x, y)
        dataloader = DataLoader(ds, batch_size=2)

        # Case1: Trace and quantize
        openvino_model = InferenceOptimizer.trace(model, accelerator='openvino', input_sample=x)
        optimized_model = InferenceOptimizer.quantize(openvino_model, accelerator='openvino',
                                                      calib_data=dataloader)
        y_hat = optimized_model(x[0:3])
        assert y_hat.shape == (3, 10)
        y_hat = optimized_model(x)
        assert y_hat.shape == (10, 10)

        # Case2: Quantize directly from pytorch
        optimized_model = InferenceOptimizer.quantize(model, accelerator='openvino',
                                                      calib_data=dataloader)

        y_hat = optimized_model(x[0:3])
        assert y_hat.shape == (3, 10)
        y_hat = optimized_model(x)
        assert y_hat.shape == (10, 10)

        trainer.validate(optimized_model, dataloader)
        trainer.test(optimized_model, dataloader)
        trainer.predict(optimized_model, dataloader)

    def test_quantize_openvino_with_tuning(self):
        trainer = Trainer()
        model = mobilenet_v3_small(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)

        ds = TensorDataset(x, y)
        dataloader = DataLoader(ds, batch_size=2)

        optimized_model = InferenceOptimizer.quantize(model, accelerator='openvino',
                                                      calib_data=dataloader,
                                                      metric=F1Score('multiclass', num_classes=10))

        y_hat = optimized_model(x[0:3])
        assert y_hat.shape == (3, 10)
        y_hat = optimized_model(x)
        assert y_hat.shape == (10, 10)

        trainer.validate(optimized_model, dataloader)
        trainer.test(optimized_model, dataloader)
        trainer.predict(optimized_model, dataloader)

    def test_quantize_openvino_option(self):
        model = mobilenet_v3_small(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)

        ds = TensorDataset(x, y)
        dataloader = DataLoader(ds, batch_size=2)

        optimized_model = InferenceOptimizer.quantize(model, accelerator='openvino',
                                                      calib_data=dataloader,
                                                      openvino_config={"PERFORMANCE_HINT": "LATENCY"})

        optimized_model(x[0:1])
    
    def test_quantize_openvino_with_tensor(self):
        model = mobilenet_v3_small(num_classes=10)

        x = torch.rand((1, 3, 256, 256))
        y = torch.ones((1, ), dtype=torch.long)

        optimized_model = InferenceOptimizer.quantize(model, accelerator='openvino',
                                                      calib_data=(x, y))

        optimized_model(x[0:1])

    def test_pytorch_openvino_model_context_manager(self):
        model = mobilenet_v3_small(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)

        ds = TensorDataset(x, y)
        dataloader = DataLoader(ds, batch_size=2)

        openvino_model = InferenceOptimizer.quantize(model,
                                                     accelerator='openvino',
                                                     calib_data=dataloader,
                                                     metric=F1Score('multiclass', num_classes=10),
                                                     thread_num=2)

        with InferenceOptimizer.get_context(openvino_model):
            assert torch.get_num_threads() == 2
            y1 = openvino_model(x[0:1])
    
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(openvino_model, tmp_dir_name)
            model = InferenceOptimizer.load(tmp_dir_name)

        with InferenceOptimizer.get_context(model):
            assert torch.get_num_threads() == 2
            y2 = model(x[0:1])

    def test_pytorch_openvino_model_additional_attrs(self):
        model = mobilenet_v3_small(num_classes=10)
        # patch a attribute
        model.channels = 3
        def hello():
            print("hello world!")
        # patch a function
        model.hello = hello

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)

        ds = TensorDataset(x, y)
        dataloader = DataLoader(ds, batch_size=2)

        openvino_model = InferenceOptimizer.quantize(model,
                                                     accelerator='openvino',
                                                     calib_data=dataloader,
                                                     metric=F1Score('multiclass', num_classes=10),
                                                     thread_num=2)
        assert openvino_model.channels == 3
        openvino_model.hello()
        with pytest.raises(
            AttributeError,
            match="'PytorchOpenVINOModel' object has no attribute 'width'"
        ):
            openvino_model.width

        with InferenceOptimizer.get_context(openvino_model):
            assert torch.get_num_threads() == 2
            y1 = openvino_model(x[0:1])

        # save & load without original model
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(openvino_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, device='CPU')
        with pytest.raises(
            AttributeError,
            match="'PytorchOpenVINOModel' object has no attribute 'channels'"
        ):
            load_model.channels
        with pytest.raises(AttributeError):
            load_model.hello()

        # save & load with original model
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(openvino_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model=model, device='CPU')
        assert load_model.channels == 3
        load_model.hello()
        with pytest.raises(
            AttributeError,
            match="'PytorchOpenVINOModel' object has no attribute 'width'"
        ):
            openvino_model.width

    def test_openvino_quantize_dynamic_axes(self):
        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.AvgPool2d(kernel_size=3, stride=3)

            def forward(self, x):
                return self.pool(x)

        model = CustomModel()
        x1 = torch.rand(1, 3, 14, 14)
        x2 = torch.rand(4, 3, 14, 14)
        x3 = torch.rand(1, 3, 12, 12)

        accmodel = InferenceOptimizer.quantize(model,
                                               accelerator="openvino",
                                               calib_data=torch.rand(1, 3, 14, 14))
        accmodel(x1)
        accmodel(x2)
        try:
            accmodel(x3)
        except Exception as e:
            assert e

        accmodel = InferenceOptimizer.quantize(model,
                                               accelerator="openvino",
                                               calib_data=torch.rand(1, 3, 14, 14),
                                               dynamic_axes={"x": [0, 2, 3]})
        accmodel(x1)
        accmodel(x2)
        accmodel(x3)

    def test_quantize_openvino_bf16(self):
        model = mobilenet_v3_small(num_classes=10)

        x = torch.rand((10, 3, 256, 256))

        try:
            optimized_model = InferenceOptimizer.quantize(model,
                                                          accelerator='openvino',
                                                          input_sample=x,
                                                          precision='bf16')
        except RuntimeError as e:
            assert e.__str__() == "Platform doesn't support BF16 format"
            return

        with InferenceOptimizer.get_context(optimized_model):
            y_hat = optimized_model(x[0:3])
            assert y_hat.shape == (3, 10)
            y_hat = optimized_model(x)
            assert y_hat.shape == (10, 10)
        
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(optimized_model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(new_model):
            y_hat = new_model(x[0:3])
            assert y_hat.shape == (3, 10)
            assert new_model.ov_model.additional_config["INFERENCE_PRECISION_HINT"] == "bf16"

    def test_openvino_gpu_quantize(self):
        # test whether contains GPU
        from openvino.runtime import Core
        core = Core()
        devices = core.available_devices
        gpu_avaliable = any('GPU' in x for x in devices)
        model = mobilenet_v3_small(num_classes=10)

        x = torch.rand((1, 3, 256, 256))
        x2 = torch.rand((10, 3, 256, 256))

        # test dynamic_shape
        with pytest.raises(RuntimeError,
                           match="For model has dynamic axes, if you want to inference on non-CPU device, must define "
                                 "input_shape for model optimizer. For more details about model optimizer, you can see mo --help ."):
            openvino_model = InferenceOptimizer.quantize(model,
                                                         input_sample=x,
                                                         accelerator='openvino',
                                                         precision='fp16',
                                                         device='GPU')
        with pytest.raises(RuntimeError,
                           match="For model has dynamic axes, if you want to inference on non-CPU device, must define "
                                 "input_shape for model optimizer. For more details about model optimizer, you can see mo --help ."):
            openvino_model = InferenceOptimizer.quantize(model,
                                                         input_sample=x,
                                                         accelerator='openvino',
                                                         dynamic_axes={'x': [0]},
                                                         precision='fp16',
                                                         device='GPU')

        if gpu_avaliable is False:
            with pytest.raises(RuntimeError):
                openvino_model = InferenceOptimizer.quantize(model,
                                                             input_sample=x,
                                                             accelerator='openvino',
                                                             device='GPU',
                                                             input_shape='[1,3,256,256]',
                                                             precision='fp16')
            return

        # test GPU fp16
        openvino_model = InferenceOptimizer.quantize(model,
                                                     input_sample=x,
                                                     accelerator='openvino',
                                                     device='GPU',
                                                     precision='fp16')
        result = openvino_model(x)
        # GPU don't support dynamic shape
        with pytest.raises(RuntimeError):
            openvino_model(x2)

        # test GPU int8
        openvino_model = InferenceOptimizer.quantize(model,
                                                     calib_data=x,
                                                     accelerator='openvino',
                                                     device='GPU',
                                                     precision='int8')
        result = openvino_model(x)
        # GPU don't support dynamic shape
        with pytest.raises(RuntimeError):
            openvino_model(x2)

    def test_openvino_vpu_quatize(self):
        # test whether contains VPU
        from openvino.runtime import Core
        core = Core()
        devices = core.available_devices
        vpu_avaliable = any('VPUX' in x for x in devices)
        
        if vpu_avaliable is False:
            return

        model = mobilenet_v3_small(num_classes=10)

        x = torch.rand((1, 3, 256, 256))
        x2 = torch.rand((10, 3, 256, 256))

        # test VPU int8
        openvino_model = InferenceOptimizer.quantize(model,
                                                     calib_data=x,
                                                     input_sample=x,
                                                     accelerator='openvino',
                                                     device='VPUX',
                                                     precision='int8')
        result = openvino_model(x)
        # VPU don't support dynamic shape
        with pytest.raises(RuntimeError):
            openvino_model(x2)
        
        # test VPU fp16
        model = resnet50()
        with pytest.raises(RuntimeError):
            openvino_model = InferenceOptimizer.quantize(model,
                                                         input_sample=x,
                                                         accelerator='openvino',
                                                         device='VPUX',
                                                         precision='fp16')

        openvino_model = InferenceOptimizer.quantize(model,
                                                     input_sample=x,
                                                     accelerator='openvino',
                                                     device='VPUX',
                                                     precision='fp16',
                                                     mean_value=[123.68,116.78,103.94])  # first type of mean value
        result = openvino_model(x)

        openvino_model = InferenceOptimizer.quantize(model,
                                                     input_sample=x,
                                                     accelerator='openvino',
                                                     device='VPUX',
                                                     precision='fp16',
                                                     mean_value="x[123.68,116.78,103.94]") # the second type of mean value
        result = openvino_model(x)

        # VPU don't support dynamic shape
        with pytest.raises(RuntimeError):
            openvino_model(x2)

    def test_openvino_kwargs(self):
        # test export kwargs and mo kwargs for openvino
        model = resnet50()
        x = torch.randn(1, 3, 224, 224)

        ov_model = InferenceOptimizer.quantize(model,
                                               accelerator="openvino",
                                               input_sample=x,
                                               calib_data=x,
                                               do_constant_folding=False, # onnx export param
                                               mean_value=[123.68,116.78,103.94]  # ov mo param
                                               )
        with InferenceOptimizer.get_context(ov_model):
            result = ov_model(x)

    def test_quantize_openvino_fp16(self):
        model = mobilenet_v3_small(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        x2 = torch.rand((1, 3, 256, 256))

        ov_fp16_model = InferenceOptimizer.quantize(model,
                                                    accelerator='openvino',
                                                    input_sample=x,
                                                    precision='fp16')

        with InferenceOptimizer.get_context(ov_fp16_model):
            preds1 = ov_fp16_model(x).numpy()
            ov_fp16_model(x2).numpy()  # test dinamic shape
        
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(ov_fp16_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)

        with InferenceOptimizer.get_context(load_model):
            preds2 = load_model(x).numpy()

        np.testing.assert_almost_equal(preds1, preds2, decimal=5)

    def test_openvino_quantize_output_tensors(self):
        model = mobilenet_v3_small(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)

        ds = TensorDataset(x, y)
        dataloader = DataLoader(ds, batch_size=2)

        openvino_model = InferenceOptimizer.quantize(model, accelerator='openvino',
                                                     calib_data=dataloader)
        test_openvino_model = InferenceOptimizer.quantize(model, accelerator='openvino',
                                                          calib_data=dataloader,
                                                          output_tensors=False)
        for x, y in dataloader:
            forward_model_tensor = openvino_model(x).numpy()
            forward_model_numpy = test_openvino_model(x)
            assert isinstance(forward_model_numpy, np.ndarray)
            np.testing.assert_almost_equal(forward_model_tensor, forward_model_numpy, decimal=5)
