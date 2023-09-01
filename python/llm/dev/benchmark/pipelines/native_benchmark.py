import torch
import os
import time
import argparse
from bigdl.llm.transformers import AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizer, AutoTokenizer
from benchmark_util import BenchmarkWrapper
from torch.profiler import profile, record_function, ProfilerActivity
import psutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Native INT4 example')
    parser.add_argument('--repo-id', type=str, required=True,
                        help='The huggingface repo id for the large language model to be downloaded'
                             ', or the path to the huggingface checkpoint folder.')
    parser.add_argument('--save-path', type=str, required=True,
                        help='path to save the model.')
    parser.add_argument('--local-model-hub', type=str,
                        help='the local caching hub dir.')
    parser.add_argument('--prompt-len', type=int, default=32,
                        help='prompt length to infer.')
    parser.add_argument('--infer-times', type=int, default=1,
                        help='inference times for tests.')
    parser.add_argument('--max-tokens', type=int, default=32,
                        help='max tokens to generate.')
    parser.add_argument('--model-family', type=str, default="llama",
                        help='The model family of the loaded model.')
    args = parser.parse_args()

    repo_id = args.repo_id
    save_path = args.save_path
    model_family = args.model_family
    max_tokens = args.max_tokens
    n_ctx = args.prompt_len + max_tokens

    if args.local_model_hub:
        repo_model_name = repo_id.split("/")[1]
        model_path = args.local_model_hub + os.path.sep + repo_model_name
    else:
        model_path = repo_id
    infer_times = args.infer_times
    from bigdl.llm import llm_convert
    bigdl_llm_path = llm_convert(model=model_path,
        outfile=save_path, outtype='int4', model_family=model_family)
    print(f"savepath={save_path}, llmconv={bigdl_llm_path}")

    #load the converted model
    from bigdl.llm.transformers import BigdlNativeForCausalLM
    model = BigdlNativeForCausalLM.from_pretrained(bigdl_llm_path, n_ctx=n_ctx)

    st = time.time()
    # TODO: auto detect using config.json
    # tokenizer = LlamaTokenizer.from_pretrained(model_path)

    end = time.time()
    print(">> loading and conversion of model costs {}s".format(end-st))

    input_str = open(f"prompt/{args.prompt_len}.txt", 'r').read()

    for i in range(infer_times):
        input_ids = model.tokenize(input_str)
        # As different tokenizer has different encodings,
        # slice the input_ids to ensure the prompt length is required length.
        print("Origin input_ids is", len(input_ids))
        input_ids = input_ids[:args.prompt_len]
        print("Sliced input length is : ", len(input_ids))
        true_input = model.batch_decode(input_ids)
        st = time.time()
        output = model(true_input, max_tokens=32)
        # output = model.batch_decode(output_ids)
        print(true_input + output['choices'][0]['text'])
        end = time.time()
