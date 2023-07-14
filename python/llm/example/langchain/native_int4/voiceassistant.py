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

# This would makes sure Python is aware there is more than one sub-package within bigdl,
# physically located elsewhere.
# Otherwise there would be module not found error in non-pip's setting as Python would
# only search the first bigdl package and end up finding only one sub-package.

# Code adapted from https://python.langchain.com/docs/use_cases/chatbots/voice_assistant


from langchain import LLMChain, PromptTemplate
from bigdl.llm.langchain.llms import BigdlNativeLLM
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import speech_recognition as sr
import pyttsx3
import argparse
import time


def prepare_chain(args):

    model_path = args.model_path
    model_family = args.model_family
    n_threads = args.thread_num

    # Use a easy prompt could bring good-enough result
    template = """
    {history}
    Q: {human_input}
    A:"""
    prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

    # We use our BigdlNativeLLM to subsititute OpenAI web-required API
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
                               
    llm = BigdlNativeLLM(
            model_path=model_path,
            model_family=model_family,
            n_threads=n_threads,
            callback_manager=callback_manager, 
            verbose=True
        )

    # Following code are complete the same as the use-case
    voiceassitant_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=2),
    )
    return voiceassitant_chain


def listen(voiceassitant_chain):


    from bigdl.llm.transformers import AutoModelForSpeechSeq2Seq
    from transformers import WhisperProcessor
    import numpy as np
    model_path = "./whisper-tiny.en"
    processor = WhisperProcessor.from_pretrained(model_path)
    recogn_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, load_in_4bit=True)
    recogn_model.config.forced_decoder_ids = None


    engine = pyttsx3.init()
    r = sr.Recognizer()
    with sr.Microphone(device_index=1, sample_rate=16000) as source:
        print("Calibrating...")
        r.adjust_for_ambient_noise(source, duration=5)
        # optional parameters to adjust microphone sensitivity
        # r.energy_threshold = 200
        # r.pause_threshold=0.5

        print("Okay, go!")
        while 1:
            text = ""
            print("listening now...")
            try:
                audio = r.listen(source, timeout=5, phrase_time_limit=30)
                # refer to https://github.com/openai/whisper/blob/main/whisper/audio.py#L63
                frame_data = np.frombuffer(audio.frame_data, np.int16).flatten().astype(np.float32) / 32768.0
                print("Recognizing...")
                st = time.time()
                input_features = processor(frame_data,
                                           sampling_rate=audio.sample_rate,
                                           return_tensors="pt").input_features
                predicted_ids = recogn_model.generate(input_features)
                text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                time_0 = time.time() - st

            except Exception as e:
                unrecognized_speech_text = (
                    f"Sorry, I didn't catch that. Exception was: \n {e}"
                )
                text = unrecognized_speech_text
            # print(text)
            st = time.time()
            response_text = voiceassitant_chain.predict(human_input=text,
                                                        stop="\n")
            print(response_text)
            print(f"Recognized in {time_0}s, Predicted in {time.time() - st}s")
            engine.say(response_text)
            engine.runAndWait()


def main(args):
    chain = prepare_chain(args)
    listen(chain)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BigDL-LLM Langchain Voice Assistant Example')
    parser.add_argument('-x','--model-family', type=str, required=True,
                        help='the model family')
    parser.add_argument('-m','--model-path', type=str, required=True,
                        help='the path to the converted llm model')
    parser.add_argument('-t','--thread-num', type=int, default=2,
                        help='Number of threads to use for inference')
    args = parser.parse_args()

    main(args)