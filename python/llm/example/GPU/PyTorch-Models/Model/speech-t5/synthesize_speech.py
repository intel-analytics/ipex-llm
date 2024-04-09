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
# Some parts of this file is adapted from
# https://huggingface.co/microsoft/speecht5_tts
#
#    MIT License
#
#    Copyright (c) Microsoft Corporation.
#
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE


import torch
import time
import argparse

from ipex_llm import optimize_model
from transformers import SpeechT5Processor, SpeechT5HifiGan, SpeechT5ForTextToSpeech
from datasets import load_dataset
import soundfile as sf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synthesize speech with the given input text using SpeechT5 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default='microsoft/speecht5_tts',
                        help='The huggingface repo id for the SpeechT5 model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--repo-id-or-vocoder-path', type=str, default='microsoft/speecht5_hifigan',
                        help='The huggingface repo id for the vocoder model (which generates audio from a spectrogram) to be downloaded'
                             ', or the path to the huggingface checkpoint folder.')
    parser.add_argument('--repo-id-or-data-path', type=str, default="Matthijs/cmu-arctic-xvectors",
                        help='The huggingface repo id for the audio dataset (which decides voice characteristics) to be downloaded'
                            ', or the path to the huggingface dataset folder')
    parser.add_argument('--text', type=str, default="Artificial intelligence refers to the development of computer systems that can perform tasks that typically require human intelligence.",
                        help='Text to synthesize speech')
    
    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    vocoder_path = args.repo_id_or_vocoder_path
    dataset_path = args.repo_id_or_data_path
    text = args.text

    processor = SpeechT5Processor.from_pretrained(model_path)
    model = SpeechT5ForTextToSpeech.from_pretrained(model_path)
    vocoder = SpeechT5HifiGan.from_pretrained(vocoder_path)

    # With only one line to enable IPEX-LLM optimization on model
    # Skip optimizing these two modules to get higher audio quality
    # When running LLMs on Intel iGPUs for Windows users, we recommend setting `cpu_embedding=True` in the optimize_model function.
    # This will allow the memory-intensive embedding layer to utilize the CPU instead of iGPU.
    model = optimize_model(model, modules_to_not_convert=["speech_decoder_postnet.feat_out",
                                                          "speech_decoder_postnet.prob_out"]) 
    model = model.to('xpu')
    vocoder = vocoder.to('xpu')

    inputs = processor(text=text, return_tensors="pt").to('xpu')
    
    # load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset(dataset_path, split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to('xpu')
    
    with torch.inference_mode():
        # ipex_llm model needs a warmup, then inference time can be accurate
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

        st = time.time()
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        torch.xpu.synchronize()
        end = time.time()
        print(f"Inference time: {end-st} s")
    
    sf.write("bigdl_llm_speech_t5_out.wav", speech.to('cpu').numpy(), samplerate=16000)
