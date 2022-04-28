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
from unittest import TestCase
from _train_torch_lightning import create_data_loader, data_transform
from bigdl.nano.pytorch.trainer import Trainer
from torchvision.models.mobilenetv3 import mobilenet_v3_small

batch_size = 256
num_workers = 0
data_dir = os.path.join(os.path.dirname(__file__), "data")


class TestPytorchQuantizedModel(TestCase):

    def test_trainer_quantize_inc_ptq_compiled(self):
        model = mobilenet_v3_small(num_classes=10)
        train_loader = create_data_loader(data_dir, batch_size, num_workers, data_transform)
        trainer = Trainer(max_epochs=1)

        qmodel = trainer.quantize(model, train_loader)
        assert qmodel
        # save/load state dict
        qmodel.load_state_dict(qmodel.state_dict())
