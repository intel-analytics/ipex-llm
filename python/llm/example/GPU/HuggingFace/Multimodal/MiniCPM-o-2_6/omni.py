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
import math
import torch
import librosa
import argparse
import numpy as np
from PIL import Image
from moviepy import VideoFileClip
from transformers import AutoTokenizer
from ipex_llm.transformers import AutoModel


# The video chunk function is adpated from https://huggingface.co/openbmb/MiniCPM-o-2_6#chat-inference
def get_video_chunk_content(video_path, temp_audio_name, flatten=True):
    video = VideoFileClip(video_path)
    print('video_duration:', video.duration)
    
    with open(temp_audio_name, 'wb') as temp_audio_file:
        temp_audio_file_path = temp_audio_file.name
        video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=16000)
        audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
    num_units = math.ceil(video.duration)
    
    # 1 frame + 1s audio chunk
    contents= []
    for i in range(num_units):
        frame = video.get_frame(i+1)
        image = Image.fromarray((frame).astype(np.uint8))
        audio = audio_np[sr*i:sr*(i+1)]
        if flatten:
            contents.extend(["<unit>", image, audio])
        else:
            contents.append(["<unit>", image, audio])
    
    return contents


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chat with MiniCPM-o-2_6 in Omni mode')
    parser.add_argument('--repo-id-or-model-path', type=str, default="openbmb/MiniCPM-o-2_6",
                        help='The Hugging Face or ModelScope repo id for the MiniCPM-o-2_6 model to be downloaded'
                             ', or the path to the checkpoint folder')
    parser.add_argument('--video-path', type=str, required=True,
                        help='The path to the video, which the model uses to conduct inference '
                             'based on its images and audio.')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')
    
    args = parser.parse_args()

    model_path = args.repo_id_or_model_path
    video_path = args.video_path

    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    model = AutoModel.from_pretrained(model_path, 
                                      load_in_low_bit="sym_int4",
                                      optimize_model=True,
                                      trust_remote_code=True,
                                      attn_implementation='sdpa',
                                      use_cache=True,
                                      init_vision=True,
                                      init_audio=True,
                                      init_tts=False,
                                      modules_to_not_convert=["apm", "vpm", "resampler"])
    
    model = model.half().to('xpu')

    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    

    # The following code for generation is adapted from https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct#quickstart
    temp_audio_name = "temp_audio.wav"
    contents = get_video_chunk_content(video_path, temp_audio_name)
    messages = [{"role":"user", "content": contents}]

    if os.path.exists(temp_audio_name):
        os.remove(temp_audio_name)

    with torch.inference_mode():
        # ipex_llm model needs a warmup, then inference time can be accurate
        model.chat(
            msgs=messages,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.5,
            max_new_tokens=args.n_predict,
            omni_input=True, # need to set omni_input=True when omni inference
            use_tts_template=False,
            generate_audio=False,
            max_slice_nums=1,
            use_image_id=False,
        )

        st = time.time()
        response = model.chat(
            msgs=messages,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.5,
            max_new_tokens=args.n_predict,
            omni_input=True, # need to set omni_input=True when omni inference
            use_tts_template=False,
            generate_audio=False,
            max_slice_nums=1,
            use_image_id=False,
        )
        torch.xpu.synchronize()
        end = time.time()

    print(f'Inference time: {end-st} s')
    print('-'*20, 'Input Video Path', '-'*20)
    print(video_path)
    print('-'*20, 'Chat Output', '-'*20)
    print(response)
