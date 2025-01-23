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
import time
import torch
import librosa
import argparse
from PIL import Image
from transformers import AutoTokenizer
from ipex_llm.transformers import AutoModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chat with MiniCPM-o-2_6 with text/audio/image')
    parser.add_argument('--repo-id-or-model-path', type=str, default="openbmb/MiniCPM-o-2_6",
                        help='The Hugging Face or ModelScope repo id for the MiniCPM-o-2_6 model to be downloaded'
                             ', or the path to the checkpoint folder')
    parser.add_argument('--image-path', type=str,
                        help='The path to the image for inference.')
    parser.add_argument('--audio-path', type=str,
                        help='The path to the audio for inference.')
    parser.add_argument('--prompt', type=str,
                        help='Prompt for inference.')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')
    
    args = parser.parse_args()

    model_path = args.repo_id_or_model_path
    image_path = args.image_path
    audio_path = args.audio_path

    modules_to_not_convert = []
    init_vision = False
    init_audio = False
    if image_path is not None and os.path.exists(image_path):
        init_vision = True
        modules_to_not_convert += ["vpm", "resampler"]
    if audio_path is not None and os.path.exists(audio_path):
        init_audio = True
        modules_to_not_convert += ["apm"]

    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    model = AutoModel.from_pretrained(model_path, 
                                      load_in_low_bit="sym_int4",
                                      optimize_model=True,
                                      trust_remote_code=True,
                                      attn_implementation='sdpa',
                                      use_cache=True,
                                      init_vision=init_vision,
                                      init_audio=init_audio,
                                      init_tts=False,
                                      modules_to_not_convert=modules_to_not_convert)
    
    model = model.half().to('xpu')

    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    

    # The following code for generation is adapted from 
    # https://huggingface.co/openbmb/MiniCPM-o-2_6#addressing-various-audio-understanding-tasks and
    # https://huggingface.co/openbmb/MiniCPM-o-2_6#chat-with-single-image
    content = []
    if init_vision:
        image_input = Image.open(image_path).convert('RGB')
        content.append(image_input)
    if args.prompt is not None:
        content.append(args.prompt)
    if init_audio:
        audio_input, _ = librosa.load(audio_path, sr=16000, mono=True)
        content.append(audio_input)
    messages = [{'role': 'user', 'content': content}]


    with torch.inference_mode():
        # ipex_llm model needs a warmup, then inference time can be accurate
        model.chat(
            msgs=messages,
            tokenizer=tokenizer,
            sampling=True,
            max_new_tokens=args.n_predict,
        )

        st = time.time()
        response = model.chat(
            msgs=messages,
            tokenizer=tokenizer,
            sampling=True,
            max_new_tokens=args.n_predict,
        )
        torch.xpu.synchronize()
        end = time.time()

    print(f'Inference time: {end-st} s')
    print('-'*20, 'Input Image Path', '-'*20)
    print(image_path)
    print('-'*20, 'Input Audio Path', '-'*20)
    print(audio_path)
    print('-'*20, 'Input Prompt', '-'*20)
    print(args.prompt)
    print('-'*20, 'Chat Output', '-'*20)
    print(response)

