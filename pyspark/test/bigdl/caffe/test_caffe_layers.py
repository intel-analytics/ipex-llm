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

from bigdl.nn.layer import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
import numpy as np
import pytest
import tempfile
from numpy import random
from numpy.testing import assert_allclose

import caffe
from caffe_layers import testlayers


class TestCaffeLayers():

    def test_caffe_layers(self):
        temp = tempfile.mkdtemp()
        for testlayer in testlayers:
            name = testlayer.name
            definition = testlayer.definition
            shapes = testlayer.shapes
            prototxtfile = temp + name + ".prototxt"
            weightfile = temp + name + ".caffemodel"
            prototxt = open(prototxtfile, 'w')
            prototxt.write(definition)
            prototxt.close()
            caffe.set_mode_cpu()
            caffe.set_random_seed(100)
            net = caffe.Net(prototxtfile, caffe.TEST)
            inputs = []
            for shape in shapes:
                (inputName, size) = shape.items()[0]
                input = random.uniform(size=size)
                net.blobs[inputName].data[...] = input
                inputs.append(input)
            cafferesult = net.forward().get(name)
            net.save(weightfile)
            model = Model.load_caffe_model(prototxtfile, weightfile, bigdl_type="float")
            model.set_seed(100)
            if len(inputs) == 1:
                inputs = inputs[0]
            bigdlResult = model.forward(inputs)
            print cafferesult
            print bigdlResult
            assert_allclose(cafferesult, bigdlResult, atol=1e-4, rtol=0)

if __name__ == "__main__":
    pytest.main([__file__])
