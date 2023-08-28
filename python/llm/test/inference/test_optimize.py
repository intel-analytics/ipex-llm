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


import unittest
import os
import pytest
import time
import torch
from bigdl.llm import optimize_model
        
class TestTransformersAPI(unittest.TestCase):

    def setUp(self):        
        thread_num = os.environ.get('THREAD_NUM')
        if thread_num is not None:
            self.n_threads = int(thread_num)
        else:
            self.n_threads = 2
    
    def test_optimize_whisper(self):
        import whisper
        model = whisper.load_model("medium")
        model = optimize_model(model, load_in_low_bit="sym_int4", optimize_atten=False)
        result = model.transcribe("extracted_audio.wav", verbose=True, language="English")
        print(result["text"])