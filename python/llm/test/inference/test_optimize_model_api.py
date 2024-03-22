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
from ipex_llm import optimize_model
        
class TestOptimizeAPI(unittest.TestCase):

    def setUp(self):   
        thread_num = os.environ.get('THREAD_NUM')
        if thread_num is not None:
            self.n_threads = int(thread_num)
        else:
            self.n_threads = 2
    
    def test_optimize_whisper(self):
        # dataset_path = os.environ.get('COMMON_VOICE_PATH')
        # reservation_audio = os.path.join(dataset_path,'reservation.mp3')
        import whisper
        model = whisper.load_model("tiny")
        model = optimize_model(model, low_bit="sym_int4", optimize_llm=False)
        # result = model.transcribe(reservation_audio, verbose=True, language="English")
        # assert "Reservation" or "reservation" in result["text"]
        
        
if __name__ == '__main__':
    pytest.main([__file__])