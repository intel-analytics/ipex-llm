from bigdl.llm.llamaindex.llms import BigdlLLM
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

llm = BigdlLLM(
        model_name="/mnt/disk1/models/Llama-2-7b-chat-hf",
        tokenizer_name="/mnt/disk1/models/Llama-2-7b-chat-hf",
        context_window=3900,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.7, "do_sample": False},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        device_map="cpu",
    )
res = llm.complete(prompt="What is AI?")
