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

import scipy
import time
import argparse

from TTS.tts.configs.bark_config import BarkConfig
from TTS.tts.models.bark import Bark
from ipex_llm import optimize_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synthesize speech with the given input text using Bark model')
    parser.add_argument('--model-path', type=str, required=True,
                        help='The local path to the Bark model checkpoint folder')
    parser.add_argument('--text', type=str, default="This is an example text for synthesize speech.",
                        help='Text to synthesize speech')

    args = parser.parse_args()
    model_path = args.model_path
    
    # Load model
    config = BarkConfig()
    model = Bark.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path, eval=True)
    
    # With only one line to enable IPEX-LLM optimization on model
    model = optimize_model(model)

    # Synthesize speech with the given input
    text = args.text
    st = time.time()
    output_dict = model.synthesize(text, config, speaker_id="random", voice_dirs=None) # with random speaker
    end = time.time()
    print(f'Time cost: {end-st} s')

    # Save the speech as a .wav file using scipy
    sampling_rate = model.config.sample_rate
    scipy.io.wavfile.write("bark_out.wav", rate=sampling_rate, data=output_dict["wav"].squeeze())
