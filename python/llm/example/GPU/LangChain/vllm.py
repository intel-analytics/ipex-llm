from ipex_llm.langchain.vllm.vllm import VLLM
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import argparse

def main(args):
    llm = VLLM(
        model=args.model_path,
        trust_remote_code=True,  # mandatory for hf models
        max_new_tokens=128,
        top_k=10,
        top_p=0.95,
        temperature=0.8,
        max_model_len=2048,
        enforce_eager=True,
        load_in_low_bit=args.load_in_low_bit,
        device="xpu",
        tensor_parallel_size=args.tensor_parallel_size,
    )

    print(llm.invoke("What is the capital of France?"))

    template = """Question: {question}

    Answer: Let's think step by step."""""
    prompt = PromptTemplate.from_template(template)

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    question = "Who was the US president in the year the first Pokemon game was released?"

    print(llm_chain.invoke(question))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Langchain integrated vLLM example')
    parser.add_argument('-m','--model-path', type=str, required=True,
                        help='the path to transformers model')
    parser.add_argument('-q', '--question', type=str, default='What is the capital of France?', help='qustion you want to ask.')
    parser.add_argument('-t', '--max-tokens', type=int, default=128, help='max tokens to generate')
    parser.add_argument('-p', '--tensor-parallel-size', type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument('-l', '--load-in-low-bit', type=str, default='sym_int4', help="low bit format")
    args = parser.parse_args()

    main(args)

