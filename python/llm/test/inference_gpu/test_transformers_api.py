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
import tempfile

import torch
from ipex_llm.transformers import AutoModelForCausalLM, AutoModel, AutoModelForSpeechSeq2Seq
from transformers import LlamaTokenizer, AutoTokenizer

device = os.environ['DEVICE']
print(f'Running on {device}')

@pytest.mark.parametrize('prompt, answer', [
    ('What is the capital of France?\n\n', 'Paris')
    ])
@pytest.mark.parametrize('Model, Tokenizer, model_path',[
    (AutoModelForCausalLM, LlamaTokenizer, os.environ.get('LLAMA2_7B_ORIGIN_PATH')),
    (AutoModel, AutoTokenizer, os.environ.get('CHATGLM2_6B_ORIGIN_PATH')),
    (AutoModelForCausalLM, AutoTokenizer, os.environ.get('MPT_7B_ORIGIN_PATH')),
    # (AutoModelForCausalLM, AutoTokenizer, os.environ.get('MISTRAL_7B_INSTRUCT_V0_1_ORIGIN_PATH')),
    # (AutoModelForCausalLM, AutoTokenizer, os.environ.get('BAICHUAN2_7B_ORIGIN_PATH')),
    # (AutoModelForCausalLM, AutoTokenizer, os.environ.get('QWEN_7B_ORIGIN_PATH')), # qwen requires transformers<4.37.0
    ])
def test_completion(Model, Tokenizer, model_path, prompt, answer):
    with torch.inference_mode():
        tokenizer = Tokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = Model.from_pretrained(model_path,
                                    load_in_4bit=True,
                                    optimize_model=True,
                                    trust_remote_code=True)
        model = model.to(device)

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output = model.generate(input_ids, max_new_tokens=32)
        model.to('cpu')   # deallocate gpu memory
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)

        assert answer in output_str

@pytest.mark.parametrize('prompt, answer', [
    ('What is the capital of France?\n\n', 'Paris')
    ])
@pytest.mark.parametrize('Model, Tokenizer, model_path',[
    (AutoModelForCausalLM, LlamaTokenizer, os.environ.get('LLAMA2_7B_ORIGIN_PATH')),
    (AutoModel, AutoTokenizer, os.environ.get('CHATGLM2_6B_ORIGIN_PATH')),
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
            loaded_model = loaded_model.to(device)

            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            output = loaded_model.generate(input_ids, max_new_tokens=32)
            loaded_model.to('cpu')   # deallocate gpu memory
            output_str = tokenizer.decode(output[0], skip_special_tokens=True)

            assert answer in output_str

def test_transformers_auto_model_for_speech_seq2seq_int4():
    with torch.inference_mode():
        from transformers import WhisperProcessor
        from datasets import load_from_disk
        model_path = os.environ.get('WHISPER_TINY_ORIGIN_PATH')
        dataset_path = os.environ.get('SPEECH_DATASET_PATH')
        processor = WhisperProcessor.from_pretrained(model_path)
        ds = load_from_disk(dataset_path)
        sample = ds[0]["audio"]
        input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features
        input_features = input_features.to(device)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, trust_remote_code=True, load_in_4bit=True, optimize_model=True)
        model = model.to(device)
        predicted_ids = model.generate(input_features)
        # decode token ids to text
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
        model.to('cpu')
        print('Output:', transcription)
        assert 'Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.' in transcription[0]


if __name__ == '__main__':
    pytest.main([__file__])
