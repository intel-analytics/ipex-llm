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
import tempfile
from unittest import TestCase
import shutil

from ipex_llm import llm_convert
from ipex_llm.transformers import AutoModelForCausalLM
from ipex_llm.optimize import optimize_model, load_low_bit, low_memory_init


llama_model_path = os.environ.get('LLAMA_ORIGIN_PATH')
gptneox_model_path = os.environ.get('GPTNEOX_ORIGIN_PATH')
bloom_model_path = os.environ.get('BLOOM_ORIGIN_PATH')
starcoder_model_path = os.environ.get('STARCODER_ORIGIN_PATH')
output_dir = os.environ.get('INT4_CKPT_DIR')

class TestConvertModel(TestCase):
    
    def test_convert_llama(self):
        converted_model_path = llm_convert(model=llama_model_path,
                                           outfile=output_dir,
                                           model_family='llama',
                                           model_format="pth",
                                           outtype='int4')
        assert os.path.isfile(converted_model_path)

    def test_convert_gptneox(self):
        converted_model_path = llm_convert(model=gptneox_model_path,
                                           outfile=output_dir,
                                           model_family='gptneox',
                                           model_format="pth",
                                           outtype='int4')
        assert os.path.isfile(converted_model_path)

    def test_convert_bloom(self):
        converted_model_path = llm_convert(model=bloom_model_path,
                                           outfile=output_dir,
                                           model_family='bloom',
                                           model_format="pth",
                                           outtype='int4')
        assert os.path.isfile(converted_model_path)
    
    def test_convert_starcoder(self):
        converted_model_path = llm_convert(model=starcoder_model_path,
                                           outfile=output_dir,
                                           model_family='starcoder',
                                           model_format="pth",
                                           outtype='int4')
        assert os.path.isfile(converted_model_path)

    def test_transformer_convert_llama(self):
        with tempfile.TemporaryDirectory(dir=output_dir) as tempdir:
            model = AutoModelForCausalLM.from_pretrained(llama_model_path, load_in_4bit=True)
            model.save_low_bit(tempdir)
            newModel = AutoModelForCausalLM.load_low_bit(tempdir)
            assert newModel is not None

    def test_transformer_convert_llama_q5(self):
        model = AutoModelForCausalLM.from_pretrained(llama_model_path,
                                                     load_in_low_bit="sym_int5")

    def test_transformer_convert_llama_q8(self):
        model = AutoModelForCausalLM.from_pretrained(llama_model_path,
                                                     load_in_low_bit="sym_int8")

    def test_transformer_convert_llama_save_load(self):
        with tempfile.TemporaryDirectory(dir=output_dir) as tempdir:
            model = AutoModelForCausalLM.from_pretrained(llama_model_path,
                                                        load_in_low_bit="asym_int4")
            model.save_low_bit(tempdir)
            newModel = AutoModelForCausalLM.load_low_bit(tempdir)
            assert newModel is not None

    def test_optimize_transformers_llama(self):
        from transformers import AutoModelForCausalLM as AutoCLM
        with tempfile.TemporaryDirectory(dir=output_dir) as tempdir:
            model = AutoCLM.from_pretrained(llama_model_path,
                                            torch_dtype="auto",
                                            low_cpu_mem_usage=True,
                                            trust_remote_code=True)
            model = optimize_model(model)
            model.save_low_bit(tempdir)
            with low_memory_init():
                new_model = AutoCLM.from_pretrained(tempdir,
                                                torch_dtype="auto",
                                                trust_remote_code=True)
            new_model = load_low_bit(new_model,
                                model_path=tempdir)
            assert new_model is not None

if __name__ == '__main__':
    pytest.main([__file__])
