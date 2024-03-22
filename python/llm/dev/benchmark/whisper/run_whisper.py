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

from datasets import load_dataset
from ipex_llm.transformers import AutoModelForSpeechSeq2Seq
from transformers import WhisperProcessor
import torch
from evaluate import load
import time
import argparse
import pandas as pd
import os
import csv
from datetime import date

current_dir = os.path.dirname(os.path.realpath(__file__))
 
def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Whisper performance and accuracy")
    parser.add_argument('--model_path', required=True, help='pretrained model path')
    parser.add_argument('--data_type', required=True, help='clean, other')
    parser.add_argument('--device', required=False, help='cpu, xpu')
    parser.add_argument('--load_in_low_bit', default='sym_int4', help='Specify whether to load data in low bit format (e.g., 4-bit)')
    parser.add_argument('--save_result', action='store_true', help='Save the results to a CSV file')
 
    args = parser.parse_args()
    return args
 
if __name__ == '__main__':
    args = get_args()
    if args.device == "":
        args.device = "cpu"
 
    speech_dataset = load_dataset('./librispeech_asr.py', name=args.data_type, split='test').select(range(500))
    processor = WhisperProcessor.from_pretrained(args.model_path)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language='en', task='transcribe')
   
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_path, load_in_low_bit=args.load_in_low_bit, optimize_model=True).eval().to(args.device)
    model.config.forced_decoder_ids = None
   
    def map_to_pred(batch):
        audio = batch["audio"]
        start_time = time.time()
        input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
        batch["reference"] = processor.tokenizer._normalize(batch['text'])
 
        with torch.no_grad():
            predicted_ids = model.generate(input_features.to(args.device), forced_decoder_ids=forced_decoder_ids, use_cache=True)[0]
            if args.device == "xpu":
                torch.xpu.synchronize()
 
        infer_time = time.time() - start_time
        transcription = processor.decode(predicted_ids)
        batch["prediction"] = processor.tokenizer._normalize(transcription)
        batch["length"] = len(audio["array"])/audio["sampling_rate"]
        batch["time"] = infer_time
        print(batch["reference"])
        print(batch["prediction"])
        return batch
   
    result = speech_dataset.map(map_to_pred, keep_in_memory=True)
    wer = load("./wer")
    speech_length = sum(result["length"][1:])
    prc_time = sum(result["time"][1:])

    MODEL = args.model_path.split('/')[-2]
    RTF = prc_time/speech_length
    RTX = speech_length/prc_time
    WER = 100 * wer.compute(references=result["reference"], predictions=result["prediction"])

    today = date.today()
    if args.save_result:
        csv_name = f'{current_dir}/results/{MODEL}-{args.data_type}-{args.device}-{args.load_in_low_bit}-{today}.csv'
        os.makedirs(os.path.dirname(csv_name), exist_ok=True)
        with open(csv_name, mode='a', newline='') as file:
            csv_writer = csv.writer(file)
            file.seek(0, os.SEEK_END)
            if file.tell() == 0:
                csv_writer.writerow(["models","precision","WER","RTF"])
            csv_writer.writerow([MODEL, args.load_in_low_bit, WER, RTF])
        print(f'Results saved to {csv_name}')

    print("Realtime Factor(RTF) is : %.4f" % RTF)
    print("Realtime X(RTX) is : %.2f" % RTX)
    print(f'WER is {WER}')