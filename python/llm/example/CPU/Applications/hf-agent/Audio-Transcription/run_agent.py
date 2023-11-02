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

from datasets import load_dataset
from transformers import AutoTokenizer, LocalAgent, WhisperProcessor
from transformers.tools.base import PipelineTool

from bigdl.llm.transformers import AutoModelForSpeechSeq2Seq, AutoModelForCausalLM


class CustomSpeechToTextTool(PipelineTool):
    default_checkpoint = "openai/whisper-tiny"
    description = (
        "This is a custom tool that transcribes an audio into text. It takes an input "
        "named `audio` and returns the transcribed text."
    )
    name = "transcriber"
    pre_processor_class = WhisperProcessor
    model_class = AutoModelForSpeechSeq2Seq

    inputs = ["audio"]
    outputs = ["text"]

    def __init__(
        self,
        language,
        model=None,
        pre_processor=None,
        post_processor=None,
        device=None,
        device_map=None,
        model_kwargs=None,
        token=None,
        **hub_kwargs,
    ):
        super().__init__(
            model,
            pre_processor,
            post_processor,
            device,
            device_map,
            model_kwargs,
            token,
            **hub_kwargs,
        )
        self.language = language

    def setup(self):
        if isinstance(self.pre_processor, str):
            # Load processor
            self.pre_processor = self.pre_processor_class.from_pretrained(
                self.pre_processor, **self.hub_kwargs
            )
            self.forced_decoder_ids = self.pre_processor.get_decoder_prompt_ids(
                language=self.language, task="transcribe"
            )

        if isinstance(self.model, str):
            # Load model in 4 bit,
            # which convert the relevant layers in the model into INT4 format
            self.model = self.model_class.from_pretrained(
                self.model, load_in_4bit=True, **self.model_kwargs, **self.hub_kwargs
            )

        super().setup()

    def encode(self, audio):
        return self.pre_processor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
        ).input_features

    def forward(self, inputs):
        # if your selected model is capable of utilizing previous key/value attentions
        # to enhance decoding speed, but has `"use_cache": false` in its model config,
        # it is important to set `use_cache=True` explicitly in the `generate` function
        # to obtain optimal performance with BigDL-LLM INT4 optimizations
        return self.model.generate(
            inputs=inputs, forced_decoder_ids=self.forced_decoder_ids
        )

    def decode(self, outputs):
        return self.pre_processor.batch_decode(outputs, skip_special_tokens=True)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run agent using vicuna model")
    parser.add_argument("--repo-id-or-model-path", type=str, default="lmsys/vicuna-7b-v1.5",
                        help="The huggingface repo id for the Vicuna model to be downloaded"
                             ", or the path to the huggingface checkpoint folder")
    parser.add_argument('--repo-id-or-data-path', type=str,
                        default="hf-internal-testing/librispeech_asr_dummy",
                        help='The huggingface repo id for the audio dataset to be downloaded'
                             ', or the path to the huggingface dataset folder')
    parser.add_argument('--language', type=str, default="english",
                        help='language to be transcribed')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    dataset_path = args.repo_id_or_data_path
    language = args.language

    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=True)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Load custom tool
    additional_tool = {"transcriber": CustomSpeechToTextTool(language)}
    # Create an agent
    agent = LocalAgent(model, tokenizer, additional_tools=additional_tool)

    # Load dummy dataset and read audio files
    ds = load_dataset(dataset_path, "clean", split="validation")
    audio = ds[0]["audio"]

    # Generate results
    prompt = "Generate the text for the 'audio'"
    print('==', 'Prompt', '==')
    print(prompt)
    print(agent.run(prompt, audio=audio))
