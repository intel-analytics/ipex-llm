from bigdl.llm.ggml.model.llama import Llama

model = Llama("model/ggml/gpt4all-model-q4_0.bin")
response=model("what is ai")
print(response)