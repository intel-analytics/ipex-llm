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


import whisper
import time
import librosa
import argparse
from ipex_llm import optimize_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recognize Tokens using `transcribe()` API for Openai Whisper model')
    parser.add_argument('--model-name', type=str, default="tiny",
                        help="The model name(tiny, medium, base, etc.) for the Whisper model to be downloaded."
                             "It is one of the official model names listed by `whisper.available_models()`, or"
                             "path to a model checkpoint containing the model dimensions and the model state_dict.")
    parser.add_argument('--audio-file', type=str, required=True,
                        help='The path of the audio file to be recognized.')
    parser.add_argument('--language', type=str, default="English",
                        help='language to be transcribed')
    args = parser.parse_args()

    # Load the input audio
    y, sr = librosa.load(args.audio_file)

    # Downsample the audio to 16kHz
    target_sr = 16000
    audio = librosa.resample(y,
                            orig_sr=sr,
                            target_sr=target_sr)

    # Load whisper model under pytorch framework
    model = whisper.load_model(args.model_name)

    # With only one line to enable IPEX-LLM optimize on a pytorch model
    model = optimize_model(model)
    
    model = model.to('xpu')

    st = time.time()
    result = model.transcribe(audio, verbose=True, language=args.language)
    end = time.time()
    print(f'Inference time: {end-st} s')

    print('-'*20, 'Output', '-'*20)
    print(result["text"])
