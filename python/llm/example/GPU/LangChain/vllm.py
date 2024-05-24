from ipex_llm.langchain.vllm.vllm import VLLM

llm = VLLM(
    model="YOUR_MODEL_PATH",
    trust_remote_code=True,  # mandatory for hf models
    max_new_tokens=128,
    top_k=10,
    top_p=0.95,
    temperature=0.8,
    max_model_len=2048,
    enforce_eager=True,
    load_in_low_bit="fp8",
    device="xpu"
)

print(llm.invoke("What is the capital of France ?"))


from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

template = """Question: {question}

Answer: Let's think step by step."""""
prompt = PromptTemplate.from_template(template)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Who was the US president in the year the first Pokemon game was released?"

print(llm_chain.invoke(question))
