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

import time
import argparse

from ipex_llm import optimize_model
from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, pipeline
from transformers.models.whisper import WhisperFeatureExtractor, WhisperTokenizer


if __name__ == '__main__':
     parser = argparse.ArgumentParser(description='Recognize Long Segment using `generate()` API for Distil-Whisper model')
     parser.add_argument('--repo-id-or-model-path', type=str, default="distil-whisper/distil-large-v2",
                         help='The huggingface repo id for the Distil-Whisper model to be downloaded'
                              ', or the path to the huggingface checkpoint folder')
     parser.add_argument('--repo-id-or-data-path', type=str,
                         default="distil-whisper/librispeech_long",
                         help='The huggingface repo id for the audio dataset to be downloaded'
                              ', or the path to the huggingface dataset folder')
     parser.add_argument('--language', type=str, default="english",
                         help='language to be transcribed')
     parser.add_argument('--batch-size', type=int, default=16,
                         help='The batch_size of pipeline inference, '
                              'it usually equals of length of the audio divided by chunk-length.')
     parser.add_argument('--chunk-length', type=int, default=15,
                         help="The maximum time lengths of chuncks of sampling_rate samples used to trim"
                              "and pad longer or shorter audio sequences. Default to be 30s.")

     args = parser.parse_args()

     model_path = args.repo_id_or_model_path
     dataset_path = args.repo_id_or_data_path

     # Load dummy dataset and read audio files
     dataset = load_dataset(dataset_path, "clean", split="validation")
     audio = dataset[0]["audio"]

     model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path)
     model = optimize_model(model)
     model.config.forced_decoder_ids = None

     pipe = pipeline(
          "automatic-speech-recognition",
          model=model,
          feature_extractor=WhisperFeatureExtractor.from_pretrained(model_path),
          tokenizer= WhisperTokenizer.from_pretrained(model_path, language=args.language),
          chunk_length_s=args.chunk_length,
     )

     start = time.time()
     prediction = pipe(audio, batch_size=args.batch_size)["text"]
     print(f"inference time is {time.time()-start}")

     print(prediction)
