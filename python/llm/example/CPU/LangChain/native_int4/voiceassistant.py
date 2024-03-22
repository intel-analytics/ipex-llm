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
from ipex_llm.langchain.llms import *
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import speech_recognition as sr
import pyttsx3
import argparse


def prepare_chain(args):

    model_path = args.model_path
    model_family = args.model_family
    n_threads = args.thread_num
    n_ctx = args.context_size

    # Use a easy prompt could bring good-enough result
    # You could tune the prompt based on your own model to perform better
    template = """
    {history}

    Q: {human_input}
    A:"""
    prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

    # We use our BigDLCausalLLM to subsititute OpenAI web-required API
    model_family_to_llm = {
        "llama": LlamaLLM,
        "gptneox": GptneoxLLM,
        "bloom": BloomLLM,
        "starcoder": StarcoderLLM,
        "chatglm": ChatGLMLLM
    }

    if model_family in model_family_to_llm:
        langchain_llm = model_family_to_llm[model_family]
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = langchain_llm(
            model_path=model_path,
            n_threads=n_threads,
            callback_manager=callback_manager,
            verbose=True,
            n_ctx=n_ctx,
            stop=['\n\n'] # You could tune the stop words based on your own model to perform better
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
    engine = pyttsx3.init()
    r = sr.Recognizer()
    with sr.Microphone() as source:
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
                print("Recognizing...")
                # whisper model options are found here: https://github.com/openai/whisper#available-models-and-languages
                # other speech recognition models are also available.
                text = r.recognize_whisper(
                    audio,
                    model="medium.en",
                    show_dict=True,
                )["text"]
            except Exception as e:
                unrecognized_speech_text = (
                    f"Sorry, I didn't catch that. Exception was: {e}s"
                )
                text = unrecognized_speech_text
            print(text)

            response_text = voiceassitant_chain.predict(human_input=text)
            print(response_text)
            engine.say(response_text)
            engine.runAndWait()


def main(args):
    chain = prepare_chain(args)
    listen(chain)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BigDLCausalLM Langchain Voice Assistant Example')
    parser.add_argument('-x','--model-family', type=str, required=True,
                        choices=["llama", "bloom", "gptneox", "chatglm", "starcoder"],
                        help='the model family')
    parser.add_argument('-m','--model-path', type=str, required=True,
                        help='the path to the converted llm model')
    parser.add_argument('-t','--thread-num', type=int, default=2,
                        help='Number of threads to use for inference')
    parser.add_argument('-c','--context-size', type=int, default=512,
                        help='Maximum context size')
    args = parser.parse_args()

    main(args)
