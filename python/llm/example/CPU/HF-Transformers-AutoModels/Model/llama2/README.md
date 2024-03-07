# Llama2
In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on Llama2 models. For illustration purposes, we utilize the [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) and [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) as reference Llama2 models.

## 0. Requirements
To run these examples with BigDL-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a Llama2 model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm

pip install bigdl-llm[all] # install bigdl-llm with 'all' option
```

### 2. Run
```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Llama2 model (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'meta-llama/Llama-2-7b-chat-hf'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

> **Note**: When loading the model in 4-bit, BigDL-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the Llama2 model based on the capabilities of your machine.

#### 2.1 Client
On client Windows machine, it is recommended to run directly with full utilization of all cores:
```powershell
python ./generate.py 
```

#### 2.2 Server
For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# set BigDL-LLM env variables
source bigdl-llm-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./generate.py
```

#### 2.3 Sample Output
#### [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
### HUMAN:
What is AI?

### RESPONSE:

-------------------- Output --------------------
### HUMAN:
What is AI?

### RESPONSE:

AI is a term used to describe the development of computer systems that can perform tasks that typically require human intelligence, such as understanding natural language, recognizing images
```

#### [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
### HUMAN:
What is AI?

### RESPONSE:

-------------------- Output --------------------
### HUMAN:
What is AI?

### RESPONSE:

AI, or artificial intelligence, refers to the ability of machines to perform tasks that would typically require human intelligence, such as learning, problem-solving,
```

### 3. (Optional) Use BigDL-IPEX Optimizations for BF16 and INT4 

To do generation with BigDL-IPEX optimizations, you need to set environment varibility `BIGDL_OPT_IPEX=true`, and config properly for `model`.

#### 3.1 For BF16 model

```python
model = AutoModelForCausalLM.from_pretrained(model_path,
                                            optimize_model=True,
                                            torch_dtype=torch.bfloat16,
                                            low_cpu_mem_usage=True, 
                                            load_in_low_bit='bf16',
                                            torchscript=True,
                                            trust_remote_code=True,
                                            use_cache=True)
```

#### 3.2 For INT4 model

*Currently verified with [main branch of IPEX](https://github.com/intel/intel-extension-for-pytorch/tree/main)*

Need to generate IPEX INT4 model (GPTQ formatted) and IPEX Quant model by following commands refering to [here](https://github.com/intel/intel-extension-for-pytorch/tree/main/examples/cpu/inference/python/llm#4116-run-in-weight-only-quantization-int4-with-ipexllm):
```bash
python ./run.py -m <your_model_path> --max-new-tokens 128 --ipex-weight-only-quantization --benchmark --input-tokens 1024 --num-warmup 1 --num-iter 3 --token-latency --greedy --weight-dtype INT4 --gptq --quant-with-amp --output-dir <your_output_path> --batch-size 1
```

You will find `gptq_checkpoint_g128.pt` for param `ipex_gptq_int4_model_path` and `best_model.pt` for param `ipex_best_model_path` in `<your_output_path>`. Then you can do generation with model configured like:

```python
model = AutoModelForCausalLM.from_pretrained(model_path,
                                            optimize_model=True,
                                            torch_dtype='auto',
                                            low_cpu_mem_usage=True, 
                                            load_in_low_bit='sym_int4',
                                            torchscript=True,
                                            trust_remote_code=True,
                                            ipex_gptq_int4_model_path=args.ipex_gptq_int4_model_path,
                                            ipex_best_model_path=args.ipex_best_model_path,
                                            use_cache=True)
```