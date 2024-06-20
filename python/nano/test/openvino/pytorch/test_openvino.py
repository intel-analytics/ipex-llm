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
from typing import Dict, Tuple, List


class TupleInputModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(28 * 28, 128)
        self.layer_3 = nn.Linear(256, 1)
    
    def forward(self, x1, x2, x3):
        x1 = self.layer_1(x1)
        x2_ = None
        for x in x2:
            if x2_ is None:
                x2_ = self.layer_2(x)
            else:
                x2_ += self.layer_2(x)
        x = torch.cat([x1, x2_], axis=1)

        return self.layer_3(x) + x3


class MultipleInputNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(10, 1)
        self.dense2 = nn.Linear(10, 1)

    def forward(self, x1, x2):
        return self.dense1(x1) + 2 * self.dense2(x2)
    
class MultipleInputNet_multi_input_type(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(12, 1)
        self.dense2 = nn.Linear(11, 1)
        self.dense3 = nn.Linear(10, 1)
        self.dense4 = nn.Linear(9, 1)
        self.dense5 = nn.Linear(8, 1)
        self.dense6 = nn.Linear(7, 1)
        self.dense7 = nn.Linear(6, 1)

    def forward(self, x_1: Dict, x_2: Tuple, x_3: List, bbox):
        return self.dense1(x_1['x1']) + self.dense2(x_1['x2']) + self.dense3(x_2[0]) + self.dense4(x_2[1]) + \
            self.dense5(x_3[0]) + self.dense6(x_3[1]) + self.dense7(bbox)


class MultipleInputNet_single_dict(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(12, 1)
        self.dense2 = nn.Linear(11, 1)
        self.dense3 = nn.Linear(10, 1)

    def forward(self, x: Dict):
        return self.dense1(x['x1']) + self.dense2(x['x2']) + self.dense3(x['x3'])
    

class MultipleInputNet_non_list_dict(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(12, 1)
        self.dense2 = nn.Linear(11, 1)
        self.dense3 = nn.Linear(10, 1)
        self.dense4 = nn.Linear(9, 1)

    def forward(self, bbox: torch.Tensor, x: Dict):
        return self.dense1(x['x1']) + self.dense2(x['x2']) + self.dense3(x['x3']) + self.dense4(bbox)


class DictOutputModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(28 * 28, 12)
        self.layer_2 = nn.Linear(28 * 28, 12)
        self.layer_3 = nn.Linear(24, 1)
    
    def forward(self, x1, x2):
        x1 = self.layer_1(x1)
        x2 = self.layer_2(x2)
        x = torch.cat([x1, x2], axis=1)
        x3 = self.layer_3(x)
        output = {"x1": x1, "x2": x2, "x3": x3}
        return output


class DictOutputModel2(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(28 * 28, 12)
        self.layer_2 = nn.Linear(28 * 28, 12)
        self.layer_3 = nn.Linear(24, 1)
    
    def forward(self, x1, x2):
        x1 = self.layer_1(x1)
        x2 = self.layer_2(x2)
        x = torch.cat([x1, x2], axis=1)
        x3 = self.layer_3(x)
        output = {"x3": x3, "x1": x1, "x2": x2}
        return output


class DictTensorOutputModel1(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(28 * 28, 12)
        self.layer_2 = nn.Linear(28 * 28, 12)
        self.layer_3 = nn.Linear(24, 1)
    
    def forward(self, x1, x2):
        x1 = self.layer_1(x1)
        x2 = self.layer_2(x2)
        x = torch.cat([x1, x2], axis=1)
        x3 = self.layer_3(x)
        output = {"x1": x1, "x2": x2}
        return output, x3


class MultiDictOutputModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(28 * 28, 12)
        self.layer_2 = nn.Linear(28 * 28, 12)
        self.layer_3 = nn.Linear(24, 1)
    
    def forward(self, x1, x2):
        x1 = self.layer_1(x1)
        x2 = self.layer_2(x2)
        x = torch.cat([x1, x2], axis=1)
        x3 = self.layer_3(x)
        output1 = {"x1": x1, "x2": x2, "x3": x3}
        output2 = {"x3": x3, "x1": x1, "x2": x2}
        return output1, output2


class MultiDictTensorOutputModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(28 * 28, 12)
        self.layer_2 = nn.Linear(28 * 28, 12)
        self.layer_3 = nn.Linear(24, 1)
    
    def forward(self, x1, x2):
        x1 = self.layer_1(x1)
        x2 = self.layer_2(x2)
        x = torch.cat([x1, x2], axis=1)
        x3 = self.layer_3(x)
        output1 = {"x1": x1, "x2": x2, "x3": x3}
        output2 = {"x3": x3, "x1": x1, "x2": x2}
        return x3, output1, output2


class ListOutputModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(28 * 28, 12)
        self.layer_2 = nn.Linear(28 * 28, 12)
        self.layer_3 = nn.Linear(24, 1)

    def forward(self, x1, x2):
        x1 = self.layer_1(x1)
        x2 = self.layer_2(x2)
        x = torch.cat([x1, x2], axis=1)
        output = self.layer_3(x)
        return [x1, x2, output]


class TupleTensorOutputModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(28 * 28, 12)
        self.layer_2 = nn.Linear(28 * 28, 12)
        self.layer_3 = nn.Linear(24, 1)
    
    def forward(self, x1, x2):
        x1 = self.layer_1(x1)
        x2 = self.layer_2(x2)
        x = torch.cat([x1, x2], axis=1)
        output = self.layer_3(x)
        return output, (x1, x2, x)


class MultiTupleOutputModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(28 * 28, 12)
        self.layer_2 = nn.Linear(28 * 28, 12)
        self.layer_3 = nn.Linear(24, 1)

    def forward(self, x1, x2):
        x1 = self.layer_1(x1)
        x2 = self.layer_2(x2)
        x = torch.cat([x1, x2], axis=1)
        output = self.layer_3(x)
        return [x1,x2], (output, x)


class MultiTupleTensorOutputModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(28 * 28, 12)
        self.layer_2 = nn.Linear(28 * 28, 12)
        self.layer_3 = nn.Linear(24, 1)
    
    def forward(self, x1, x2):
        x1 = self.layer_1(x1)
        x2 = self.layer_2(x2)
        x = torch.cat([x1, x2], axis=1)
        output = self.layer_3(x)
        return output, [x1,x2], (output, x)


class TupleDictOutputModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(28 * 28, 12)
        self.layer_2 = nn.Linear(28 * 28, 12)
        self.layer_3 = nn.Linear(24, 1)
    
    def forward(self, x1, x2):
        x1 = self.layer_1(x1)
        x2 = self.layer_2(x2)
        x = torch.cat([x1, x2], axis=1)
        output = self.layer_3(x)
        return [x1,x2], {"x":x, "output":output}


class TupleDictOutputModel2(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(28 * 28, 12)
        self.layer_2 = nn.Linear(28 * 28, 12)
        self.layer_3 = nn.Linear(24, 1)
    
    def forward(self, x1, x2):
        x1 = self.layer_1(x1)
        x2 = self.layer_2(x2)
        x = torch.cat([x1, x2], axis=1)
        output = self.layer_3(x)
        return [x1,x2,{"x":x, "output":output}]

class TupleDictOutputModel3(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(28 * 28, 12)
        self.layer_2 = nn.Linear(28 * 28, 12)
        self.layer_3 = nn.Linear(24, 1)
    
    def forward(self, x1, x2):
        x1 = self.layer_1(x1)
        x2 = self.layer_2(x2)
        x = torch.cat([x1, x2], axis=1)
        output = self.layer_3(x)
        return {"intermediate": [x1,x2], "x":x, "output":output}


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
        from datetime import datetime
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

        target_sample1 = unet(image_latents, torch.Tensor([980]).long(), encoder_hidden_states)
        target_sample2 = unet(image_latents2, torch.Tensor([980]).long(), encoder_hidden_states2)

        # Static shape
        dynamic_axes= False
        nano_unet = InferenceOptimizer.trace(unet, accelerator="openvino",
                                            input_sample=input_sample,
                                            input_names=["sample", "timestep",
                                                        "encoder_hidden_states", "return_dict"],
                                            output_names=["unet_output"],
                                            dynamic_axes=dynamic_axes,
                                            device='CPU',
                                            )
        nano_unet(image_latents, torch.Tensor([980]).long(), encoder_hidden_states)
        # Inference with wrong batchsize
        with pytest.raises(RuntimeError,
                        match="The input blob size is not equal to the network input size"):
            nano_unet(image_latents2, torch.Tensor([980]).long(), encoder_hidden_states2)

        # Test save, load, reshape(fail) & cache
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(nano_unet, tmp_dir_name)

            # Load without shapes
            first_load_start = datetime.utcnow()
            new_model = InferenceOptimizer.load(tmp_dir_name, cache_dir=tmp_dir_name)
            first_load_end = datetime.utcnow()

            # Reshape using same shape, should not recompile
            new_model.reshape("sample[2,4,8,8],encoder_hidden_states[2,12,10]")
            reshape_end = datetime.utcnow()
            assert (reshape_end - first_load_end).total_seconds() * 1000 < 50

            new_model(image_latents, torch.Tensor([980]).long(), encoder_hidden_states)
            # Inference with wrong batchsize
            with pytest.raises(RuntimeError,
                            match="The input blob size is not equal to the network input size"):
                new_model(image_latents2, torch.Tensor([980]).long(), encoder_hidden_states2)

            # Load with caching
            second_load_start = datetime.utcnow()
            new_model = InferenceOptimizer.load(tmp_dir_name, cache_dir=tmp_dir_name)
            second_load_end = datetime.utcnow()
            assert (second_load_end - second_load_start) < (first_load_end - first_load_start)

            # Reshape failed, should not recompile
            success, error_msg = new_model.reshape("sample[1,4,64,64],encoder_hidden_states[1,12,10]")
            reshape_end = datetime.utcnow()
            new_model_inputs = {i.any_name: i.shape for i in new_model.ov_model.ie_network.inputs}
            assert not success
            assert "Failed to reshape this model." in error_msg
            assert list(new_model_inputs["sample"]) == [2, 4, 8, 8]
            assert list(new_model_inputs["encoder_hidden_states"]) == [2, 12, 10]
            assert (reshape_end - second_load_end).total_seconds() * 1000 < 50

            # Load with wrong shape
            new_model = InferenceOptimizer.load(tmp_dir_name, cache_dir=tmp_dir_name, 
                                                shapes="sample[1,4,8,8],encoder_hidden_states[1,12,10]")
            new_model_inputs = {i.any_name: i.shape for i in new_model.ov_model.ie_network.inputs}
            assert list(new_model_inputs["sample"]) == [2, 4, 8, 8]
            assert list(new_model_inputs["encoder_hidden_states"]) == [2, 12, 10]

        # Dynamic shapes
        dynamic_axes= {"sample": [0],
                       "encoder_hidden_states": [0],
                       "unet_output": [0]}
        nano_unet = InferenceOptimizer.trace(unet, accelerator="openvino",
                                            input_sample=input_sample,
                                            input_names=["sample", "timestep",
                                                        "encoder_hidden_states", "return_dict"],
                                            output_names=["unet_output"],
                                            dynamic_axes=dynamic_axes,
                                            device='CPU',
                                            input_shape="[-1,4,8,8],[1],[-1,12,10]",
                                            input="sample,timestep,encoder_hidden_states"
                                            )
        output1 = nano_unet(image_latents, torch.Tensor([980]).long(), encoder_hidden_states)
        output2 = nano_unet(image_latents2, torch.Tensor([980]).long(), encoder_hidden_states2)
        np.testing.assert_almost_equal(output1.sample.detach().numpy(), target_sample1.sample.detach().numpy(), decimal=5)
        np.testing.assert_almost_equal(output2.sample.detach().numpy(), target_sample2.sample.detach().numpy(), decimal=5)

        nano_unet.reshape("sample[1,4,8,8],encoder_hidden_states[1,12,10]")
        nano_unet_inputs = {i.any_name: i.shape for i in nano_unet.ov_model.ie_network.inputs}
        assert list(nano_unet_inputs["sample"]) == [1, 4, 8, 8]
        assert list(nano_unet_inputs["encoder_hidden_states"]) == [1, 12, 10]

        # Test load with shapes
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(nano_unet, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name, 
                                                shapes="sample[1,4,8,8],encoder_hidden_states[1,12,10]")
            new_model_inputs = {i.any_name: i.shape for i in new_model.ov_model.ie_network.inputs}
            assert list(new_model_inputs["sample"]) == [1, 4, 8, 8]
            assert list(new_model_inputs["encoder_hidden_states"]) == [1, 12, 10]
        output2 = new_model(image_latents2, torch.Tensor([980]).long(), encoder_hidden_states2)
        np.testing.assert_almost_equal(output2.sample.detach().numpy(), target_sample2.sample.detach().numpy(), decimal=5)

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
        
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(openvino_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(test_openvino_model, tmp_dir_name)
            test_load_model = InferenceOptimizer.load(tmp_dir_name)

        for x, y in dataloader:
            forward_model_tensor = load_model(x).numpy()
            forward_model_numpy = test_load_model(x)
            assert isinstance(forward_model_numpy, np.ndarray)
            np.testing.assert_almost_equal(forward_model_tensor, forward_model_numpy, decimal=5)

    def test_openvino_tuple_input(self):
        model = TupleInputModel()
        x1 = torch.randn(100, 28 * 28)
        x2 = [torch.randn(100, 28 * 28), torch.randn(100, 28 * 28), torch.randn(100, 28 * 28)]  # tuple
        x3 = 5
        target = model(x1, x2, x3)

        ov_model = InferenceOptimizer.trace(model, accelerator='openvino', input_sample=(x1, x2, x3))
        with InferenceOptimizer.get_context(ov_model):
            output1 = ov_model(x1, x2, x3)
            np.testing.assert_almost_equal(target.numpy(), output1.numpy(), decimal=5)
            # test tuple input as kwargs
            output2 = ov_model(x1, x2=x2, x3=x3)
            np.testing.assert_almost_equal(output1.numpy(), output2.numpy(), decimal=5)
        
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(ov_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)
        
        with InferenceOptimizer.get_context(load_model):
            output3 = load_model(x1, x2, x3)
            np.testing.assert_almost_equal(target.numpy(), output3.numpy(), decimal=5)
            # test tuple input as kwargs
            output4 = ov_model(x1, x2=x2, x3=x3)
            np.testing.assert_almost_equal(output3.numpy(), output4.numpy(), decimal=5)

    def test_openvino_keyword_argument(self):
        net = MultipleInputNet()
        x1 = torch.randn(32, 10)
        x2 = torch.randn(32, 10)

        ov_model = InferenceOptimizer.trace(net,
                                            accelerator="openvino",
                                            input_sample=(x1,x2))
        with InferenceOptimizer.get_context(ov_model):
            output1 = ov_model(x1, x2).numpy()
            # test keyword argument
            output2 = ov_model(x1, x2=x2).numpy()
            output3 = ov_model(x1=x1, x2=x2).numpy()
            np.testing.assert_allclose(output1, output2, atol=1e-5)
            np.testing.assert_allclose(output1, output3, atol=1e-5)

            # test some bad cases
            with pytest.raises(RuntimeError):
                ov_model(x1, x2, x2=x2)

            with pytest.raises(RuntimeError):
                ov_model(x1, x2, x1=x2)

            with pytest.raises(RuntimeError):
                ov_model(x2, x1=x2)

            with pytest.raises(RuntimeError):
                ov_model(x1, x1=x2)

    def test_openvino_dict_output(self):
        x1 = torch.randn(10, 28 * 28)
        x2 = torch.randn(10, 28 * 28)
        # test1: output is a single dict
        for Model in [DictOutputModel, DictOutputModel2]:
            model = Model()
            output = model(x1, x2)
            assert isinstance(output, dict)

            ov_model = InferenceOptimizer.trace(model, accelerator='openvino', input_sample=(x1, x2))
            with InferenceOptimizer.get_context(ov_model):
                output1 = ov_model(x1, x2)

            assert output.keys() == output1.keys()
            for k in output.keys():
                np.testing.assert_almost_equal(output[k].detach().numpy(), output1[k].detach().numpy(), decimal=5)

            with tempfile.TemporaryDirectory() as tmp_dir_name:
                InferenceOptimizer.save(ov_model, tmp_dir_name)
                load_model = InferenceOptimizer.load(tmp_dir_name)

            with InferenceOptimizer.get_context(load_model):
                output2 = load_model(x1, x2)

            assert output.keys() == output2.keys()
            for k in output.keys():
                np.testing.assert_almost_equal(output[k].detach().numpy(), output2[k].detach().numpy(), decimal=5)

        # test2: output is a dict with other non-list items
        model = DictTensorOutputModel1()
        dic, out = model(x1, x2)
        assert isinstance(dic, dict)
        ov_model = InferenceOptimizer.trace(model, accelerator='openvino', input_sample=(x1, x2))
        with InferenceOptimizer.get_context(ov_model):
            dic1, out1 = ov_model(x1, x2)
        assert dic1.keys() == dic.keys()
        np.testing.assert_almost_equal(out.detach().numpy(), out1.detach().numpy(), decimal=5)
        
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(ov_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)

        with InferenceOptimizer.get_context(load_model):
            dic2, out2 = load_model(x1, x2)
        assert dic2.keys() == dic1.keys()
        np.testing.assert_almost_equal(out2.detach().numpy(), out1.detach().numpy(), decimal=5)

        # test3: test multi dict, output are 2 dicts
        model = MultiDictOutputModel()
        dic1, dic2 = model(x1, x2)
        assert isinstance(dic1, dict)
        assert isinstance(dic2, dict)
        ov_model = InferenceOptimizer.trace(model, accelerator='openvino', input_sample=(x1, x2))
        with InferenceOptimizer.get_context(ov_model):
            output_dic1, output_dic2 = ov_model(x1, x2)
        assert dic1.keys() == output_dic1.keys()
        assert dic2.keys() == output_dic2.keys()

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(ov_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)

        with InferenceOptimizer.get_context(load_model):
            output2_dic1, output2_dic2 = load_model(x1, x2)
        assert dic1.keys() == output2_dic1.keys()
        assert dic2.keys() == output2_dic2.keys()
        
        # test4: test multi dict with non-list item, output is a tensor with 2 dicts
        model = MultiDictTensorOutputModel()
        output, dic1, dic2 = model(x1, x2)
        assert isinstance(dic1, dict)
        assert isinstance(dic2, dict)
        ov_model = InferenceOptimizer.trace(model, accelerator='openvino', input_sample=(x1, x2))
        with InferenceOptimizer.get_context(ov_model):
            output1, output1_dic1, output1_dic2 = ov_model(x1, x2)
        assert dic1.keys() == output_dic1.keys()
        assert dic2.keys() == output_dic2.keys()
        np.testing.assert_almost_equal(output.detach().numpy(), output1.detach().numpy(), decimal=5)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(ov_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)

        with InferenceOptimizer.get_context(load_model):
            output2, output2_dic1, output2_dic2 = load_model(x1, x2)
        assert dic1.keys() == output2_dic1.keys()
        assert dic2.keys() == output2_dic2.keys()
        np.testing.assert_almost_equal(output.detach().numpy(), output1.detach().numpy(), decimal=5)

    def test_openvino_list_output(self):
        x1 = torch.randn(10, 28 * 28)
        x2 = torch.randn(10, 28 * 28)
        # test1: output is a single list
        model = ListOutputModel()
        output = model(x1, x2)
        assert isinstance(output, list)

        ov_model = InferenceOptimizer.trace(model, accelerator='openvino', input_sample=(x1, x2))
        with InferenceOptimizer.get_context(ov_model):
            output1 = ov_model(x1, x2)
        
        assert len(output) == len(output1)
        for k in range(len(output)):
            np.testing.assert_almost_equal(output[k].detach().numpy(), output1[k].detach().numpy(), decimal=5)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(ov_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)

        with InferenceOptimizer.get_context(load_model):
            output2 = load_model(x1, x2)

        assert len(output) == len(output2)
        for k in range(len(output)):
            np.testing.assert_almost_equal(output[k].detach().numpy(), output2[k].detach().numpy(), decimal=5)

        # test2: output is a tuple with other non-list items
        model = TupleTensorOutputModel()
        out, list_out = model(x1, x2)
        assert isinstance(list_out, tuple)
        ov_model = InferenceOptimizer.trace(model, accelerator='openvino', input_sample=(x1, x2))
        with InferenceOptimizer.get_context(ov_model):
            out1, list_out1 = ov_model(x1, x2)
        assert len(list_out) == len(list_out1)
        np.testing.assert_almost_equal(out.detach().numpy(), out1.detach().numpy(), decimal=5)
        for k in range(len(list_out)):
            np.testing.assert_almost_equal(list_out[k].detach().numpy(), list_out1[k].detach().numpy(), decimal=5)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(ov_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)

        with InferenceOptimizer.get_context(load_model):
            out2, list_out2 = load_model(x1, x2)
        assert len(list_out) == len(list_out2)
        np.testing.assert_almost_equal(out2.detach().numpy(), out1.detach().numpy(), decimal=5)
        for k in range(len(list_out1)):
            np.testing.assert_almost_equal(list_out1[k].detach().numpy(), list_out2[k].detach().numpy(), decimal=5)

        # test3: test multi list/tuple, output are 2 lists
        model = MultiTupleOutputModel()
        list1, list2 = model(x1, x2)
        assert isinstance(list1, list)
        assert isinstance(list2, tuple)
        ov_model = InferenceOptimizer.trace(model, accelerator='openvino', input_sample=(x1, x2))
        with InferenceOptimizer.get_context(ov_model):
            output_list1, output_list2 = ov_model(x1, x2)
        assert len(output_list1) == len(list1)
        assert len(output_list2) == len(list2)
        for k in range(len(list1)):
            np.testing.assert_almost_equal(list1[k].detach().numpy(), output_list1[k].detach().numpy(), decimal=5)
        for k in range(len(list2)):
            np.testing.assert_almost_equal(list2[k].detach().numpy(), output_list2[k].detach().numpy(), decimal=5)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(ov_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)

        with InferenceOptimizer.get_context(load_model):
            output2_list1, output2_list2 = load_model(x1, x2)
        assert len(output2_list1) == len(list1)
        assert len(output2_list2) == len(list2)
        for k in range(len(list1)):
            np.testing.assert_almost_equal(list1[k].detach().numpy(), output2_list1[k].detach().numpy(), decimal=5)
        for k in range(len(list2)):
            np.testing.assert_almost_equal(list2[k].detach().numpy(), output2_list2[k].detach().numpy(), decimal=5)

        # test4: test multi list/tuple with non-list item, output is a tensor with 2list/tuples
        model = MultiTupleTensorOutputModel()
        output, list1, list2 = model(x1, x2)
        assert isinstance(list1, list)
        assert isinstance(list2, tuple)
        ov_model = InferenceOptimizer.trace(model, accelerator='openvino', input_sample=(x1, x2))
        with InferenceOptimizer.get_context(ov_model):
            output1, output1_list1, output1_list2 = ov_model(x1, x2)
        assert len(output1_list1) == len(list1)
        assert len(output1_list2) == len(list2)
        for k in range(len(list1)):
            np.testing.assert_almost_equal(list1[k].detach().numpy(), output1_list1[k].detach().numpy(), decimal=5)
        for k in range(len(list2)):
            np.testing.assert_almost_equal(list2[k].detach().numpy(), output1_list2[k].detach().numpy(), decimal=5)
        np.testing.assert_almost_equal(output.detach().numpy(), output1.detach().numpy(), decimal=5)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(ov_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)

        with InferenceOptimizer.get_context(load_model):
            output2, output2_list1, output2_list2 = load_model(x1, x2)
        assert len(output2_list1) == len(list1)
        assert len(output2_list2) == len(list2)
        for k in range(len(list1)):
            np.testing.assert_almost_equal(list1[k].detach().numpy(), output2_list1[k].detach().numpy(), decimal=5)
        for k in range(len(list2)):
            np.testing.assert_almost_equal(list2[k].detach().numpy(), output2_list2[k].detach().numpy(), decimal=5)
        np.testing.assert_almost_equal(output2.detach().numpy(), output1.detach().numpy(), decimal=5)

    def test_openvino_list_dict_output(self):
        x1 = torch.randn(10, 28 * 28)
        x2 = torch.randn(10, 28 * 28)
        # test1: test single dict and a single list
        model = TupleDictOutputModel()
        output, dic = model(x1, x2)
        assert isinstance(output, list)
        assert isinstance(dic, dict)
        ov_model = InferenceOptimizer.trace(model, accelerator='openvino', input_sample=(x1, x2))
        with InferenceOptimizer.get_context(ov_model):
            output1, dic1 = ov_model(x1, x2)
        assert isinstance(output1, list)
        assert isinstance(dic1, dict)
        assert dic.keys() == dic1.keys()
        for k in range(len(output)):
            np.testing.assert_almost_equal(output[k].detach().numpy(), output1[k].detach().numpy(), decimal=5)
        for k in dic.keys():
            np.testing.assert_almost_equal(dic[k].detach().numpy(), dic1[k].detach().numpy(), decimal=5)

        # test2: test list contains dict
        model = TupleDictOutputModel2()
        output = model(x1, x2)
        assert isinstance(output, list)
        assert isinstance(output[2], dict)
        ov_model = InferenceOptimizer.trace(model, accelerator='openvino', input_sample=(x1, x2))
        with InferenceOptimizer.get_context(ov_model):
            output1 = ov_model(x1, x2)
        assert isinstance(output1, list)
        assert isinstance(output1[2], dict)
        assert len(output1) == len(output)
        for i in range(2):
            np.testing.assert_almost_equal(output[i].detach().numpy(), output1[i].detach().numpy(), decimal=5)
        assert output[2].keys() == output1[2].keys()
        for k in output[2].keys():
            np.testing.assert_almost_equal(output[2][k].detach().numpy(), output1[2][k].detach().numpy(), decimal=5)

        # test3: test dict contains list
        model = TupleDictOutputModel3()
        output = model(x1, x2)
        assert isinstance(output, dict)
        assert isinstance(output["intermediate"], list)
        ov_model = InferenceOptimizer.trace(model, accelerator='openvino', input_sample=(x1, x2))
        with InferenceOptimizer.get_context(ov_model):
            output1 = ov_model(x1, x2)
        assert isinstance(output1, dict)
        assert isinstance(output1["intermediate"], list)
        assert output1.keys() == output.keys()
        for k in output.keys():
            if k != "intermediate":
                np.testing.assert_almost_equal(output[k].detach().numpy(), output1[k].detach().numpy(), decimal=5)
            else:
                for i in range(2):
                    np.testing.assert_almost_equal(output["intermediate"][i].detach().numpy(), output1["intermediate"][i].detach().numpy(), decimal=5)

    def test_openvino_multi_input_type(self):
        x1_t = torch.randn(32, 12)
        x2_t = torch.randn(32, 11)
        x3_t = torch.randn(32, 10)
        x4_t = torch.randn(32, 9)
        x5_t = torch.randn(32, 8)
        x6_t = torch.randn(32, 7)
        bbox = torch.randn(32, 6)
        model_ft = MultipleInputNet_multi_input_type()

        with torch.no_grad():
            output0 = model_ft({'x1':x1_t, 'x2':x2_t}, (x3_t, x4_t), [x5_t, x6_t], bbox)

        sample = ({'x1':x1_t, 'x2':x2_t}, (x3_t, x4_t), [x5_t, x6_t], bbox)
        ov_model = InferenceOptimizer.trace(model_ft,
                                            accelerator='openvino',
                                            input_sample=sample)

        with InferenceOptimizer.get_context(ov_model):
            output1 = ov_model({'x1':x1_t, 'x2':x2_t}, (x3_t, x4_t), [x5_t, x6_t], bbox)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(ov_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)

        with InferenceOptimizer.get_context(load_model):
            output2 = load_model({'x1':x1_t, 'x2':x2_t}, (x3_t, x4_t), [x5_t, x6_t], bbox)

        np.testing.assert_almost_equal(output0.numpy(), output1.numpy(), decimal=5)
        np.testing.assert_almost_equal(output1.numpy(), output2.numpy(), decimal=5)

    def test_openvino_single_dict_input(self):
        x1_t = torch.randn(32, 12)
        x2_t = torch.randn(32, 11)
        x3_t = torch.randn(32, 10)
        model_ft = MultipleInputNet_single_dict()

        with torch.no_grad():
            output0 = model_ft({'x1':x1_t, 'x2':x2_t, 'x3':x3_t})

        sample = {'x1':x1_t, 'x2':x2_t, 'x3':x3_t}
        ov_model = InferenceOptimizer.trace(model_ft,
                                            accelerator='openvino',
                                            input_sample=sample)

        with InferenceOptimizer.get_context(ov_model):
            output1 = ov_model({'x1':x1_t, 'x2':x2_t, 'x3':x3_t})

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(ov_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)

        with InferenceOptimizer.get_context(load_model):
            output2 = load_model({'x1':x1_t, 'x2':x2_t, 'x3':x3_t})

        np.testing.assert_almost_equal(output0.numpy(), output1.numpy(), decimal=5)
        np.testing.assert_almost_equal(output1.numpy(), output2.numpy(), decimal=5)
    
    def test_openvino_non_list_dict_input(self):
        x1_t = torch.randn(32, 12)
        x2_t = torch.randn(32, 11)
        x3_t = torch.randn(32, 10)
        bbox = torch.randn(32, 9)
        model_ft = MultipleInputNet_non_list_dict()

        with torch.no_grad():
            output0 = model_ft(bbox, {'x1':x1_t, 'x2':x2_t, 'x3':x3_t})

        sample = (bbox, {'x1':x1_t, 'x2':x2_t, 'x3':x3_t})
        ov_model = InferenceOptimizer.trace(model_ft,
                                            accelerator='openvino',
                                            input_sample=sample)

        with InferenceOptimizer.get_context(ov_model):
            output1 = ov_model(bbox, {'x1':x1_t, 'x2':x2_t, 'x3':x3_t})

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(ov_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)

        with InferenceOptimizer.get_context(load_model):
            output2 = load_model(bbox, {'x1':x1_t, 'x2':x2_t, 'x3':x3_t})

        np.testing.assert_almost_equal(output0.numpy(), output1.numpy(), decimal=5)
        np.testing.assert_almost_equal(output1.numpy(), output2.numpy(), decimal=5)
