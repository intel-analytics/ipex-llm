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
import intel_extension_for_pytorch as ipex
from transformers import WhisperProcessor
import torch
import time
from benchmark_util import BenchmarkWrapper
from bigdl.llm.transformers import AutoModelForSpeechSeq2Seq
from datasets import load_dataset, load_from_disk

if __name__ == '__main__':
    parser = argparse.ArgumentParser('OPT generation script', add_help=False)
    parser.add_argument('-m', '--model-dir',
                        default="/mnt/disk1/models/whisper-medium", type=str)
    args = parser.parse_args()
    print(args)

    model_path = args.model_dir
    dataset_path = "hf-internal-testing/librispeech_asr_dummy"

    # load model and processor
    ds = load_dataset(dataset_path, "clean", split="validation")
    print("pass")
    processor = WhisperProcessor.from_pretrained(model_path)
    print("model loaded")
    # load dummy dataset and read audio files
    sample = ds[0]["audio"]
    # for transformer == 4.30.2
    input_features = processor(
        sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features

    input_features = input_features.half().contiguous()
    input_features = input_features.to('xpu')
    print(input_features.shape)
    print(input_features.is_contiguous())

    # generate token ids
    whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path, load_in_4bit=True, optimize_model=False)
    whisper.config.forced_decoder_ids = None
    whisper = whisper.half().to('xpu')
    whisper = BenchmarkWrapper(whisper)

    with torch.inference_mode():
        e2e_time = []
        for i in range(10):
            torch.xpu.synchronize()
            st = time.time()
            predicted_ids = whisper.generate(input_features)
            # print(len(predicted_ids[0]))
            torch.xpu.synchronize()
            output_str = processor.batch_decode(
                predicted_ids, skip_special_tokens=True)
            end = time.time()
            e2e_time.append(end-st)

        print(f'Inference time: {end-st} s')
        print('Output:', output_str)
        print(f'Inference time: {end-st} s')
        print(e2e_time)
