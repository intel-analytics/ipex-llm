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

import argparse
from io import BytesIO
from urllib.request import urlopen
import librosa
import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from ipex_llm import optimize_model

def main(args):
    model_path = args.repo_id_or_model_path
    max_length = args.max_length
    audio_url = args.audio_url

    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path)
    model = optimize_model(model, low_bit='sym_int4', optimize_llm=True)
    model = model.half().to('xpu')

    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio_url},
        ]},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(librosa.load(
                        BytesIO(urlopen(ele['audio_url']).read()),
                        sr=processor.feature_extractor.sampling_rate)[0]
                    )

    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    inputs = inputs.to('xpu')

    with torch.inference_mode():
        generate_ids = model.generate(**inputs, max_length=max_length) # warmup
        import time
        st = time.time()
        generate_ids = model.generate(**inputs, max_length=max_length)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        et = time.time()
        print(f'Inference time: {et-st} s')

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(response)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Qwen2-Audio")
    parser.add_argument('--repo-id-or-model-path', type=str, default="Qwen/Qwen2-Audio-7B-Instruct",
                        help='The huggingface repo id for the Qwen2-Audio model checkpoint')
    parser.add_argument('--max-length', type=int, default=256,
                        help='The max length of input text')
    parser.add_argument('--audio-url', type=str, default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/translate_to_chinese.wav",
                        help='The URL to the input audio file')
    args = parser.parse_args()
    main(args)
