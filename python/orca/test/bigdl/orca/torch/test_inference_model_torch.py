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
import pytest

from unittest import TestCase
from bigdl.orca.inference import InferenceModel
from bigdl.orca.torch import zoo_pickle_module
import torch
import torchvision
from bigdl.dllib.utils.nncontext import *


class TestInferenceModelTorch(TestCase):

    def setUp(self):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        self.sc = init_spark_on_local(4)

    def tearDown(self):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        self.sc.stop()

    def test_load_torch(self):
        torch_model = torchvision.models.resnet18()
        tmp_path = create_tmp_path() + ".pt"
        torch.save(torch_model, tmp_path, pickle_module=zoo_pickle_module)
        model = InferenceModel(10)
        model.load_torch(tmp_path)
        input_data = np.random.random([4, 3, 224, 224])
        output_data = model.predict(input_data)
        os.remove(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__])
