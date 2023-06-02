
from bigdl.llm.langchain.llms import BigDLLLM
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


template ="""{question}"""

prompt = PromptTemplate(template=template, input_variables=["question"])


# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Verbose is required to pass to the callback manager


# Make sure the model path is correct for your system!
llm = BigDLLLM(
    model_path="model/ggml/gpt4all-model-q4_0.bin", 
    # model_path="models/ggml/vicuna-model-q4_0.bin",
    callback_manager=callback_manager, 
    verbose=True
)


llm_chain = LLMChain(prompt=prompt, llm=llm)

#question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

question = "What is AI?"
llm_chain.run(question)