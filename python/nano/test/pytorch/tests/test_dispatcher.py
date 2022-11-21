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
import tempfile
import os
import pytest

from bigdl.nano.utils.log4Error import invalidOperationError
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10


class TestDispatcherPytorch(TestCase):

    def test_dispatch_pytorch(self):
        from bigdl.nano.pytorch import patch_torch, unpatch_torch
        patch_torch()
        import pytorch_lightning
        import bigdl.nano.pytorch
        import torchvision
        assert issubclass(pytorch_lightning.Trainer, bigdl.nano.pytorch.Trainer)
        assert torchvision.datasets is bigdl.nano.pytorch.vision.datasets
        assert torchvision.transforms is bigdl.nano.pytorch.vision.transforms

        unpatch_torch()
        assert not issubclass(pytorch_lightning.Trainer, bigdl.nano.pytorch.Trainer)
        assert not torchvision.datasets is bigdl.nano.pytorch.vision.datasets
        assert not torchvision.transforms is bigdl.nano.pytorch.vision.transforms

    def test_dispatch_cuda(self):
        from bigdl.nano.pytorch import patch_torch, unpatch_torch
        patch_torch(cuda_to_cpu=True)

        import torch

        # Tensor.cuda test
        a = torch.tensor([1,2,3])
        a.cuda()

        # torch.device
        assert torch.device('cuda:0').type == "cpu"

        # set device
        torch.cuda.set_device(0)

        # autocast
        amp = torch.cuda.amp.autocast()

        if not TORCH_VERSION_LESS_1_10:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                model = torch.nn.Linear(8, 8)
                input_tensor = torch.ones((1, 8))
                model(input_tensor)

        # GradScaler
        scaler = torch.cuda.amp.GradScaler()
        scaler.scale(a)

        # test if GradScaler could be inherited
        class GradScaler(torch.cuda.amp.GradScaler):
            pass

        # other tensor type
        tensor = torch.cuda.FloatTensor([0])
        tensor = tensor.to(device='cuda:0')
        device = torch.device('cuda:0')
        tensor = tensor.to(device)

        # tensor create
        tensor = torch.randn((10, 224, 224, 3), device=device)
        tensor = torch.zeros((10, 224, 224, 3), device=1)
        tensor = torch.tensor([0], device='cuda:0')

        # save, load
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "tensor.pt")
            torch.save(tensor, ckpt_name)
            tensor = torch.load(ckpt_name, map_location='cuda:0')

        # other utils
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

        unpatch_torch()

    def test_unpatch_cuda(self):
        from bigdl.nano.pytorch import patch_torch, unpatch_torch
        patch_torch(cuda_to_cpu=True)
        patch_torch(cuda_to_cpu=True) # call patch twice
        import torch
        unpatch_torch()

        with pytest.raises((RuntimeError, AssertionError), match="CUDA|NVIDIA"):
            _t = torch.tensor([0], device='cuda:0')
            invalidOperationError(False, "cuda unpatch is incorrect")

        cuda_device = torch.device('cuda:0')
        assert cuda_device.type == 'cuda'
