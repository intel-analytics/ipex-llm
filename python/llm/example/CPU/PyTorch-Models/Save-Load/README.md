# Save/Load Low-Bit Models with IPEX-LLM Optimizations

In this example, we show how to save/load model with IPEX-LLM low-bit optimizations (including INT8/INT5/INT4), and then run inference on the optimized low-bit model.

## 0. Requirements
To run this example with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../../README.md#system-support) for more information.

## Example: Save/Load Model in Low-Bit Optimization
In the example [generate.py](./generate.py), we show a basic use case of saving/loading model in low-bit optimizations to predict the next N tokens using `generate()` API. Also, saving and loading operations are platform-independent, so you could run it on different platforms.
### 1. Install
We suggest using conda to manage environment:

On Linux:

```bash
conda create -n llm python=3.11 # recommend to use Python 3.11
conda activate llm

# install the latest ipex-llm nightly build with 'all' option
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
```

On Windows:

```cmd
conda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all]
```

### 2. Run
If you want to save the optimized low-bit model, run:
```
python ./generate.py --save-path path/to/save/model
```

If you want to load the optimized low-bit model, run:
```
python ./generate.py --load-path path/to/load/model
```

In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Llama2 model (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'meta-llama/Llama-2-7b-chat-hf'`.
- `--low-bit`: argument defining the low-bit optimization data type, options are sym_int4, asym_int4, sym_int5, asym_int5 or sym_int8. (sym_int4 means symmetric int 4, asym_int4 means asymmetric int 4, etc.). Relevant low bit optimizations will be applied to the model.
- `--save-path`: argument defining the path to save the low-bit model. Then you can load the low-bit directly.
- `--load-path`: argument defining the path to load low-bit model.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

### 3 Sample Output
#### [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
```log
Inference time: xxxx s
-------------------- Output --------------------
### HUMAN:
What is AI?

### RESPONSE:

AI is a term used to describe the development of computer systems that can perform tasks that typically require human intelligence, such as understanding natural language, recognizing images
```

#### [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
```log
Inference time: xxxx s
-------------------- Output --------------------
### HUMAN:
What is AI?

### RESPONSE:

AI, or artificial intelligence, refers to the ability of machines to perform tasks that would typically require human intelligence, such as learning, problem-solving,
```