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

import torch
import scipy
import time
import argparse

from bigdl.llm import optimize_model
from transformers import AutoProcessor, BarkModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synthesize speech with the given input text using Bark model')
    parser.add_argument('--repo-id-or-model-path', type=str, default='suno/bark-small',
                        help='The huggingface repo id for the Bark model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--text', type=str, default="This is an example text for synthesize speech.",
                        help='Text to synthesize speech')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    text = args.text
    
    # Load model
    processor = AutoProcessor.from_pretrained(model_path)
    model = BarkModel.from_pretrained(model_path)
    
    # With only one line to enable BigDL-LLM optimization on model
    # When running LLMs on Intel iGPUs for Windows users, we recommend setting `cpu_embedding=True` in the optimize_model function.
    # This will allow the memory-intensive embedding layer to utilize the CPU instead of iGPU.
    model = optimize_model(model, cpu_embedding=True)
    model = model.to('xpu')

    voice_preset = "v2/en_speaker_6"
    inputs = processor(text, voice_preset=voice_preset).to('xpu')

    with torch.inference_mode():
        # ipex model needs a warmup, then inference time can be accurate
        audio_array = model.generate(**inputs)

        st = time.time()
        audio_array = model.generate(**inputs)
        torch.xpu.synchronize()
        end = time.time()

        print(f"Inference time: {end-st} s")
 
        audio_array = audio_array.cpu().numpy().squeeze()
 
        from scipy.io.wavfile import write as write_wav
        sample_rate = model.generation_config.sample_rate
        write_wav("bark_out.wav", sample_rate, audio_array)
