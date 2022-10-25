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
