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
import torch
import time
import argparse
import numpy as np

from bigdl.llm.transformers import AutoModelForCausalLM
from bigdl.llm.transformers import AutoModelForSpeechSeq2Seq
from transformers import LlamaTokenizer
import intel_extension_for_pytorch as ipex
from transformers import WhisperProcessor
from transformers import TextStreamer
from colorama import Fore
import speech_recognition as sr
from datasets import load_dataset


# you could tune the prompt based on your own model,
# here the prompt tuning refers to https://huggingface.co/georgesung/llama2_7b_chat_uncensored#prompt-style
DEFAULT_SYSTEM_PROMPT = """\
"""

def get_prompt(message: str, chat_history: list[tuple[str, str]],
               system_prompt: str) -> str:
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    return ''.join(texts)

def get_input_features(r):
    with sr.Microphone(device_index=1, sample_rate=16000) as source:
        print("Calibrating...")
        r.adjust_for_ambient_noise(source, duration=5)

        print(Fore.YELLOW + "Listening now..." + Fore.RESET)
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=30)
            # refer to https://github.com/openai/whisper/blob/main/whisper/audio.py#L63
            frame_data = np.frombuffer(audio.frame_data, np.int16).flatten().astype(np.float32) / 32768.0
            input_features = processor(frame_data, sampling_rate=audio.sample_rate, return_tensors="pt").input_features
            input_features = input_features.half().contiguous().to('xpu')
            print("Recognizing...")
        except Exception as e:
            unrecognized_speech_text = (
                f"Sorry, I didn't catch that. Exception was: \n {e}"
            )
            print(unrecognized_speech_text)
    
    return input_features

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--llama2-repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--whisper-repo-id-or-model-path', type=str, default="openai/whisper-small",
                        help='The huggingface repo id for the Whisper (e.g. `openai/whisper-small` and `openai/whisper-medium`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')

    args = parser.parse_args()
    whisper_model_path = args.whisper_repo_id_or_model_path
    llama_model_path = args.llama2_repo_id_or_model_path

    dataset_path = "hf-internal-testing/librispeech_asr_dummy"
    # Load dummy dataset and read audio files
    ds = load_dataset(dataset_path, "clean", split="validation")

    print("Converting and loading models...")
    processor = WhisperProcessor.from_pretrained(whisper_model_path)

    # generate token ids
    whisper =  AutoModelForSpeechSeq2Seq.from_pretrained(whisper_model_path, load_in_4bit=True, optimize_model=False)
    whisper.config.forced_decoder_ids = None
    whisper = whisper.half().to('xpu')
    
    llama_model = AutoModelForCausalLM.from_pretrained(llama_model_path, load_in_4bit=True, trust_remote_code=True, optimize_model=False)
    llama_model = llama_model.half().to('xpu')
    tokenizer = LlamaTokenizer.from_pretrained(llama_model_path)

    r = sr.Recognizer()

    with torch.inference_mode():
        # warm up
        sample = ds[2]["audio"]
        input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features
        input_features = input_features.half().contiguous().to('xpu')
        torch.xpu.synchronize()
        predicted_ids = whisper.generate(input_features)
        torch.xpu.synchronize()
        output_str = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        output_str = output_str[0]
        input_ids = tokenizer.encode(output_str, return_tensors="pt").to('xpu')
        output = llama_model.generate(input_ids, do_sample=False, max_new_tokens=32)
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        torch.xpu.synchronize()
        
        while 1:
            input_features = get_input_features(r)
            predicted_ids = whisper.generate(input_features)
            output_str = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            output_str = output_str[0]
            print("\n" + Fore.GREEN + "Whisper : " + Fore.RESET + "\n" + output_str)
            print("\n" + Fore.BLUE + "BigDL-LLM: " + Fore.RESET)
            prompt = get_prompt(output_str, [], system_prompt=DEFAULT_SYSTEM_PROMPT)
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu')
            streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
            _ = llama_model.generate(input_ids, streamer=streamer, do_sample=False, max_new_tokens=args.n_predict)
