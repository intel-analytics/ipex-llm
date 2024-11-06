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
import tempfile
import time
import torch
import pytest

from ipex_llm.transformers import AutoModel, AutoModelForCausalLM, AutoModelForSpeechSeq2Seq
from transformers import AutoTokenizer, LlamaTokenizer

class TestTransformersAPI(unittest.TestCase):

    def setUp(self):        
        thread_num = os.environ.get('THREAD_NUM')
        if thread_num is not None:
            self.n_threads = int(thread_num)
        else:
            self.n_threads = 2

    def test_transformers_auto_model_int4(self):
        model_path = os.environ.get('ORIGINAL_CHATGLM2_6B_PATH')
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, load_in_4bit=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        input_str = "Tell me the capital of France.\n\n"

        with torch.inference_mode():
            st = time.time()
            input_ids = tokenizer.encode(input_str, return_tensors="pt")
            output = model.generate(input_ids, do_sample=False, max_new_tokens=32)
            output_str = tokenizer.decode(output[0], skip_special_tokens=True)
            end = time.time()
        print('Prompt:', input_str)
        print('Output:', output_str)
        print(f'Inference time: {end-st} s')
        res = 'Paris' in output_str
        self.assertTrue(res)

    def test_transformers_auto_model_for_causal_lm_int4(self):
        model_path = os.environ.get('ORIGINAL_CODESHELL_7B_PATH')
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        input_str = 'def hello():\n  print("hello world")\n'
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, load_in_4bit=True)
        with torch.inference_mode():

            st = time.time()
            input_ids = tokenizer.encode(input_str, return_tensors="pt")
            output = model.generate(input_ids, do_sample=False, max_new_tokens=32)
            output_str = tokenizer.decode(output[0], skip_special_tokens=True)
            end = time.time()
        print('Prompt:', input_str)
        print('Output:', output_str)
        print(f'Inference time: {end-st} s')
        res = '\nhello()' in output_str
        self.assertTrue(res)
        

    def test_transformers_auto_model_for_speech_seq2seq_int4(self):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        from datasets import load_from_disk
        model_path = os.environ.get('ORIGINAL_WHISPER_TINY_PATH')
        dataset_path = os.environ.get('SPEECH_DATASET_PATH')
        processor = WhisperProcessor.from_pretrained(model_path)
        ds = load_from_disk(dataset_path)
        sample = ds[0]["audio"]
        input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, trust_remote_code=True, load_in_4bit=True)
        with torch.inference_mode():
            st = time.time()
            predicted_ids = model.generate(input_features)
            # decode token ids to text
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
            end = time.time()
        print('Output:', transcription)
        print(f'Inference time: {end-st} s')
        res = 'Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.' in transcription[0]
        self.assertTrue(res)

    def test_transformers_chatglm_for_causallm(self):
        from ipex_llm.transformers import ChatGLMForCausalLM
        model_path = os.environ.get('ORIGINAL_CHATGLM2_6B_PATH')
        model = ChatGLMForCausalLM.from_pretrained(model_path, native=False, trust_remote_code=True, load_in_4bit=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        input_str = "Tell me the capital of France.\n\n"

        with torch.inference_mode():
            st = time.time()
            input_ids = tokenizer.encode(input_str, return_tensors="pt")
            output = model.generate(input_ids, do_sample=False, max_new_tokens=32)
            output_str = tokenizer.decode(output[0], skip_special_tokens=True)
            end = time.time()
        print('Prompt:', input_str)
        print('Output:', output_str)
        print(f'Inference time: {end-st} s')
        res = 'Paris' in output_str
        self.assertTrue(res)

@pytest.mark.parametrize('prompt, answer', [
    ('What is the capital of France?\n\n', 'Paris')
    ])
@pytest.mark.parametrize('Model, Tokenizer, model_path',[
    (AutoModel, AutoTokenizer, os.environ.get('ORIGINAL_CHATGLM2_6B_PATH')),
    (AutoModelForCausalLM, AutoTokenizer, os.environ.get('MISTRAL_ORIGIN_PATH')),
    ])
def test_load_low_bit_completion(Model, Tokenizer, model_path, prompt, answer):
    tokenizer = Tokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = Model.from_pretrained(model_path,
                                  load_in_4bit=True,
                                  optimize_model=True,
                                  trust_remote_code=True)

    with tempfile.TemporaryDirectory() as tempdir:
        model.save_low_bit(tempdir)
        loaded_model = Model.load_low_bit(tempdir,
                                          optimize_model=True,
                                          trust_remote_code=True)

        with torch.inference_mode():
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            output = loaded_model.generate(input_ids, max_new_tokens=32)
            output_str = tokenizer.decode(output[0], skip_special_tokens=True)

            assert answer in output_str

prompt = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"

@pytest.mark.parametrize("Model, Tokenizer, model_path, prompt", [
    (AutoModelForCausalLM, LlamaTokenizer, os.environ.get('LLAMA_ORIGIN_PATH'), prompt),
    (AutoModelForCausalLM, AutoTokenizer, os.environ.get('BLOOM_ORIGIN_PATH'), prompt),
    (AutoModel, AutoTokenizer, os.environ.get('ORIGINAL_CHATGLM2_6B_PATH'), prompt),
    (AutoModelForCausalLM, AutoTokenizer, os.environ.get('ORIGINAL_CODESHELL_7B_PATH'), prompt),
    (AutoModelForCausalLM, AutoTokenizer, os.environ.get('MISTRAL_ORIGIN_PATH'), prompt)
])

def test_optimize_model(Model, Tokenizer, model_path, prompt):
    tokenizer = Tokenizer.from_pretrained(model_path, trust_remote_code=True)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.inference_mode():
        model = Model.from_pretrained(model_path,
                                      load_in_4bit=True,
                                      optimize_model=False,
                                      trust_remote_code=True)
        logits_base_model = (model(input_ids)).logits

        model = Model.from_pretrained(model_path,
                                      load_in_4bit=True,
                                      optimize_model=True,
                                      trust_remote_code=True)
        logits_optimized_model = (model(input_ids)).logits
        diff = abs(logits_base_model - logits_optimized_model).flatten()

        assert (diff/logits_base_model.flatten()).mean()<0.05


if __name__ == '__main__':
    pytest.main([__file__])
