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
from tempfile import TemporaryDirectory
from unittest import TestCase
from bigdl.nano.pytorch import Trainer
from bigdl.nano.pytorch import InferenceOptimizer
from torchvision.models.mobilenetv3 import mobilenet_v3_small
from torchvision.models import resnet50
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
import os
import tempfile
import pytest


class TestOpenVINO(TestCase):
    def test_trace_openvino(self):
        trainer = Trainer(max_epochs=1)
        model = mobilenet_v3_small(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)

        # trace a torch model
        openvino_model = InferenceOptimizer.trace(model, x, 'openvino')
        y_hat = openvino_model(x)
        assert y_hat.shape == (10, 10)

        # trace pytorch-lightning model
        pl_model = Trainer.compile(model, loss=torch.nn.CrossEntropyLoss(),
                                   optimizer=torch.optim.SGD(model.parameters(), lr=0.01))
        ds = TensorDataset(x, y)
        dataloader = DataLoader(ds, batch_size=2)
        trainer.fit(pl_model, dataloader)

        openvino_model = InferenceOptimizer.trace(model, accelerator='openvino')
        y_hat = openvino_model(x[0:3])
        assert y_hat.shape == (3, 10)
        y_hat = openvino_model(x)
        assert y_hat.shape == (10, 10)

        trainer.validate(openvino_model, dataloader)
        trainer.test(openvino_model, dataloader)
        trainer.predict(openvino_model, dataloader)

    def test_save_openvino(self):
        model = mobilenet_v3_small(num_classes=10)
        x = torch.rand((10, 3, 256, 256))

        # save and load pytorch model
        openvino_model = InferenceOptimizer.trace(model, accelerator='openvino', input_sample=x)
        with TemporaryDirectory() as saved_root:
            InferenceOptimizer.save(openvino_model, saved_root)
            assert len(os.listdir(saved_root)) > 0
            loaded_openvino_model = InferenceOptimizer.load(saved_root)
            y_hat = loaded_openvino_model(x[0:3])
            assert y_hat.shape == (3, 10)
            y_hat = loaded_openvino_model(x)
            assert y_hat.shape == (10, 10)

    def test_pytorch_openvino_model_async_predict(self):
        model = mobilenet_v3_small(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)

        openvino_model = InferenceOptimizer.trace(model, input_sample=x, accelerator='openvino')
        ds = TensorDataset(x, y)
        dataloader = DataLoader(ds, batch_size=2)

        # do async_predict use dataloader as input
        result = openvino_model.async_predict(dataloader, num_requests=3)
        for res in result:
            assert res.shape == (2, 10)

        # do async_predict use List of Tensor as input
        x = [torch.rand((10, 3, 256, 256)) for i in range(3)]
        result = openvino_model.async_predict(x, num_requests=3)
        for res in result:
            assert res.shape == (10, 10)

    def test_pytorch_openvino_model_option(self):
        model = mobilenet_v3_small(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)

        openvino_model = InferenceOptimizer.trace(model, input_sample=x,
                                                  accelerator='openvino',
                                                  openvino_config={"PERFORMANCE_HINT": "LATENCY"})

        result = openvino_model(x[0:1])

    def test_pytorch_openvino_model_context_manager(self):
        model = mobilenet_v3_small(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)

        openvino_model = InferenceOptimizer.trace(model, input_sample=x,
                                                  accelerator='openvino',
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

        openvino_model = InferenceOptimizer.trace(model, input_sample=x,
                                                  accelerator='openvino',
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
        with pytest.raises(AttributeError):
            load_model.channels == 3
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

        with InferenceOptimizer.get_context(load_model):
            assert torch.get_num_threads() == 2
            y2 = load_model(x[0:1])

    def test_openvino_default_values(self):
        # default bool values
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x, a=True, b=False):
                if a:
                    return x+1
                if b:
                    return x-1
                return x

        model = Net()

        data = torch.rand(1,3,1,1)
        result_true = model(data)
        # sample with only required parameters (in a tuple)
        accmodel = InferenceOptimizer.trace(model,
                                            accelerator="openvino",
                                            input_sample=(torch.rand(2,3,1,1),))
        result_m = accmodel(data)
        assert torch.equal(result_true, result_m)

        # sample with only required parameters
        accmodel = InferenceOptimizer.trace(model,
                                            accelerator="openvino",
                                            input_sample=torch.rand(2,3,1,1))
        result_m = accmodel(data)
        assert torch.equal(result_true, result_m)

        data = torch.rand(1,3,1,1)
        result_true = model(data, False, True)
        # sample with only required parameters
        accmodel = InferenceOptimizer.trace(model,
                                            accelerator="openvino",
                                            input_sample=(torch.rand(2,3,1,1),False,True))
        result_m = accmodel(data)
        assert torch.equal(result_true, result_m)

        # typehint model
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x: torch.Tensor, y: int = 3):
                return x+y

        model = Net()

        x = torch.rand(1,3,1,1)
        y = 3
        result_true = model(x, y)
        # sample with only required parameters (in a tuple)
        accmodel = InferenceOptimizer.trace(model,
                                            accelerator="openvino",
                                            input_sample=torch.rand(2,3,1,1))
        result_m = accmodel(x, y)
        assert torch.equal(result_true, result_m)

        # sample with only all parameters (in a tuple)
        accmodel = InferenceOptimizer.trace(model,
                                            accelerator="openvino",
                                            input_sample=(torch.rand(2,3,1,1), 3))
        result_m = accmodel(x, y)
        assert torch.equal(result_true, result_m)

    def test_openvino_dynamic_axes(self):
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

        accmodel = InferenceOptimizer.trace(model,
                                            accelerator="openvino",
                                            input_sample=torch.rand(1, 3, 14, 14))
        accmodel(x1)
        accmodel(x2)
        try:
            accmodel(x3)
        except Exception as e:
            assert e

        accmodel = InferenceOptimizer.trace(model,
                                            accelerator="openvino",
                                            input_sample=torch.rand(1, 3, 14, 14),
                                            dynamic_axes={"x": [0, 2, 3]})
        accmodel(x1)
        accmodel(x2)
        accmodel(x3)

    def test_openvino_gpu_trace(self):
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
            openvino_model = InferenceOptimizer.trace(model,
                                                      input_sample=x,
                                                      accelerator='openvino',
                                                      device='GPU')
        with pytest.raises(RuntimeError,
                           match="For model has dynamic axes, if you want to inference on non-CPU device, must define "
                                 "input_shape for model optimizer. For more details about model optimizer, you can see mo --help ."):
            openvino_model = InferenceOptimizer.trace(model,
                                                      input_sample=x,
                                                      accelerator='openvino',
                                                      dynamic_axes={'x': [0]},
                                                      device='GPU')

        if gpu_avaliable is False:
            with pytest.raises(RuntimeError):
                openvino_model = InferenceOptimizer.trace(model,
                                                          input_sample=x,
                                                          accelerator='openvino',
                                                          device='GPU',
                                                          input_shape='[1,3,256,256]')
            return

        # test GPU fp32
        openvino_model = InferenceOptimizer.trace(model,
                                                  input_sample=x,
                                                  accelerator='openvino',
                                                  device='GPU')
        result = openvino_model(x)
        assert result.shape == (1, 10)
        # GPU don't support dynamic shape
        with pytest.raises(RuntimeError):
            openvino_model(x2)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(openvino_model, tmp_dir_name)
            model = InferenceOptimizer.load(tmp_dir_name)  # GPU model
            model = InferenceOptimizer.load(tmp_dir_name, device='CPU')  # CPU model

    def test_openvino_trace_kwargs(self):
        # test export kwargs and mo kwargs for openvino
        model = resnet50()
        x = torch.randn(1, 3, 224, 224)

        ov_model = InferenceOptimizer.trace(model,
                                            accelerator="openvino",
                                            input_sample=x,
                                            do_constant_folding=False, # onnx export param
                                            mean_value=[123.68,116.78,103.94],  # ov mo param,
                                            input="x",  # test multi params for mo
                                            )
        with InferenceOptimizer.get_context(ov_model):
            result = ov_model(x)

    def test_openvino_trace_stable_diffusion_unet(self):
        from diffusers.models import UNet2DConditionModel
        # reduce model size as action runner has limited memory
        unet = UNet2DConditionModel(sample_size=64,
                                    cross_attention_dim=10,
                                    attention_head_dim=1,
                                    down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                                    block_out_channels=(32, 64),
                                    up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
                                    layers_per_block=1)
        latent_shape = (2, 4, 8, 8)
        image_latents = torch.randn(latent_shape, device = "cpu", dtype=torch.float32)
        encoder_hidden_states = torch.randn((2, 12, 10), device = "cpu", dtype=torch.float32)
        input_sample = (image_latents, torch.Tensor([980]).long(), encoder_hidden_states, False)

        latent_shape2 = (1, 4, 8, 8) # different shape
        image_latents2 = torch.randn(latent_shape2, device = "cpu", dtype=torch.float32)
        encoder_hidden_states2 = torch.randn((1, 12, 10), device = "cpu", dtype=torch.float32)

        unet(image_latents, torch.Tensor([980]).long(), encoder_hidden_states)
        unet(image_latents2, torch.Tensor([980]).long(), encoder_hidden_states2)
        
        dynamic_axes= {"sample": [0],
                       "encoder_hidden_states": [0],
                       "unet_output": [0]}
        nano_unet = InferenceOptimizer.trace(unet, accelerator="openvino",
                                             input_sample=input_sample,
                                             input_names=["sample", "timestep",
                                                          "encoder_hidden_states", "return_dict"],
                                             output_names=["unet_output"],
                                             dynamic_axes=dynamic_axes,
                                             device='CPU')
        nano_unet(image_latents, torch.Tensor([980]).long(), encoder_hidden_states)
        nano_unet(image_latents2, torch.Tensor([980]).long(), encoder_hidden_states2)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(nano_unet, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name)
        new_model(image_latents, torch.Tensor([980]).long(), encoder_hidden_states)
        new_model(image_latents2, torch.Tensor([980]).long(), encoder_hidden_states2)

    def test_openvino_trace_output_tensors(self):
        model = mobilenet_v3_small(pretrained=True)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)

        pl_model = Trainer.compile(model, loss=torch.nn.CrossEntropyLoss(),
                                   optimizer=torch.optim.Adam(model.parameters(), lr=0.01))
        ds = TensorDataset(x, y)
        dataloader = DataLoader(ds, batch_size=2)

        openvino_model = InferenceOptimizer.trace(model, accelerator='openvino', input_sample=dataloader)
        test_openvino_model = InferenceOptimizer.trace(model, accelerator='openvino',
                                                       input_sample=dataloader, output_tensors=False)

        for x, y in dataloader:
            forward_model_tensor = openvino_model(x).numpy()
            forward_model_numpy = test_openvino_model(x)
            assert isinstance(forward_model_numpy, np.ndarray)
            np.testing.assert_almost_equal(forward_model_tensor, forward_model_numpy, decimal=5)
