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
import time
import argparse

# from transformers import AutoModelForSpeechSeq2Seq
from ipex_llm.transformers.npu_model import AutoModelForSpeechSeq2Seq
from transformers import WhisperProcessor
from datasets import load_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recognize Tokens using `generate()` API for Whisper model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="D:\llm-models\whisper-tiny",
                        help='The huggingface repo id for the Whisper model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--repo-id-or-data-path', type=str,
                        default="D:\binbin\ipex-llm\librispeech_asr_dummy",
                        help='The huggingface repo id for the audio dataset to be downloaded'
                             ', or the path to the huggingface dataset folder')
    parser.add_argument('--language', type=str, default="english",
                        help='language to be transcribed')
    parser.add_argument("--max-output-len", type=int, default=1600)
    parser.add_argument("--max-prompt-len", type=int, default=1501)
    parser.add_argument("--disable-transpose-value-cache", action="store_true", default=False)
    parser.add_argument("--intra-pp", type=int, default=None)
    parser.add_argument("--inter-pp", type=int, default=None)
    
    
    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    dataset_path = args.repo_id_or_data_path
    language = args.language

    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    # model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path,
    #                                                   trust_remote_code=True,)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, 
                                                      torch_dtype=torch.float32,
                                                      # torch_dtype=torch.float16,
                                                      trust_remote_code=True,
                                                      attn_implementation="eager",
                                                      load_in_low_bit="sym_int4",
                                                      optimize_model=True,
                                                      max_output_len=args.max_output_len,
                                                      max_prompt_len=args.max_prompt_len,
                                                      intra_pp=args.intra_pp,
                                                      inter_pp=args.inter_pp,
                                                      transpose_value_cache=False,
                                                      modules_to_not_convert=['decoder', 'proj_out']
                                                      )
    print(model)
    model.config.forced_decoder_ids = None

    # Load processor
    processor = WhisperProcessor.from_pretrained(model_path)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")

    # Load dummy dataset and read audio files
    ds = load_dataset(r'D:\binbin\ipex-llm\librispeech_asr_dummy', "default", split="validation")

    # Generate predicted tokens
    with torch.inference_mode():
        sample = ds[0]["audio"]

        input_features = processor(sample["array"],
                                   sampling_rate=sample["sampling_rate"],
                                   return_tensors="pt").input_features 
        st = time.time()
        # if your selected model is capable of utilizing previous key/value attentions
        # to enhance decoding speed, but has `"use_cache": false` in its model config,
        # it is important to set `use_cache=True` explicitly in the `generate` function
        # to obtain optimal performance with IPEX-LLM INT4 optimizations
        predicted_ids = model.generate(input_features,
                                       forced_decoder_ids=forced_decoder_ids)
        end = time.time()
        output_str = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Output', '-'*20)
        print(output_str)
