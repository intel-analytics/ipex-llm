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


import pytest
import os
from unittest import TestCase

from bigdl.llm import convert_model


llama_model_path = os.environ.get('LLAMA_ORIGIN_PATH')
gptneox_model_path = os.environ.get('GPTNEOX_ORIGIN_PATH')
bloom_model_path = os.environ.get('BLOOM_ORIGIN_PATH')
output_dir = os.environ.get('INT4_CKPT_DIR')

class TestConvertModel(TestCase):
    
    def test_convert_llama(self):
        converted_model_path = convert_model(input_path=llama_model_path,
                                             output_path=output_dir,
                                             model_family='llama',
                                             model_type="pth",
                                             dtype='int4')
        assert os.path.isfile(converted_model_path)

    def test_convert_gptneox(self):
        converted_model_path = convert_model(input_path=gptneox_model_path,
                                             output_path=output_dir,
                                             model_family='gptneox',
                                             model_type="pth",
                                             dtype='int4')
        assert os.path.isfile(converted_model_path)

    def test_convert_bloom(self):
        converted_model_path = convert_model(input_path=bloom_model_path,
                                             output_path=output_dir,
                                             model_family='bloom',
                                             model_type="pth",
                                             dtype='int4')
        assert os.path.isfile(converted_model_path)

if __name__ == '__main__':
    pytest.main([__file__])
