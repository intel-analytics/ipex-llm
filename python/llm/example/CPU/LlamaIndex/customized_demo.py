from typing import Optional, List, Mapping,Any

from llama_index.core import SimpleDirectoryReader, SummaryIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata

from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core import Settings
from custom_LLM import BigdlLLM

# Transform a string into input zephyr-specific input
def completion_to_prompt(completion):
    return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"


# Transform a list of chat messages into zephyr-specific input
def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|system|>\n{message.content}</s>\n"
        elif message.role == "user":
            prompt += f"<|user|>\n{message.content}</s>\n"
        elif message.role == "assistant":
            prompt += f"<|assistant|>\n{message.content}</s>\n"

    # ensure we start with a system prompt, insert blank if needed
    if not prompt.startswith("<|system|>\n"):
        prompt = "<|system|>\n</s>\n" + prompt

    # add final assistant prompt
    prompt = prompt + "<|assistant|>\n"

    return prompt


import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings

Settings.llm = HuggingFaceLLM(
    model_name="/mnt/disk1/models/Llama-2-7b-chat-hf",
    tokenizer_name="/mnt/disk1/models/Llama-2-7b-chat-hf",
    context_window=1024,
    max_new_tokens=64,
    generate_kwargs={"temperature": 0.7, "do_sample": False},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    device_map="auto",
)

Settings.embed_model = "local:/mnt/disk1/models/bge-small-en"
Settings.llm = OurLLM(model_id="/mnt/disk1/models/Llama-2-7b-chat-hf")

# Load the your data
documents = SimpleDirectoryReader("./data").load_data()
index = SummaryIndex.from_documents(documents)

# Query and print response
query_engine = index.as_query_engine()
response = query_engine.query("<query_text>")
print(response)