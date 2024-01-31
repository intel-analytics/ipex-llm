# Phixtral
In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on phi models on [Intel GPUs](../../../README.md). For illustration purposes, we utilize the [mlabonne/phixtral-4x2_8](https://huggingface.co/mlabonne/phixtral-4x2_8) as a reference phi model.

## 0. Requirements
To run these examples with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a phi model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations on Intel GPUs.
### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
pip install einops # additional package required for phi to conduct generation
```

### 2. Configures OneAPI environment variables
```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. Run

For optimal performance on Arc, it is recommended to set several environment variables.

```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```

```
python ./generate.py --prompt 'What is AI?'
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the phi model (e.g. `mlabonne/phixtral-4x2_8`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'mlabonne/phixtral-4x2_8'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `128`.

#### Sample Output
#### [mlabonne/phixtral-4x2_8](https://huggingface.co/mlabonne/phixtral-4x2_8)

```log
Inference time: 3.88 s
-------------------- Prompt --------------------
Question: What is AI

 Answer:
-------------------- Output --------------------
Question: What is AI?

 Answer:Artificial Intelligence (AI) is a rapidly growing field that focuses on the development of intelligent machines that can perform tasks that would normally require human intelligence. AI technology is designed to mimic human intelligence, allowing machines to learn, reason, and make decisions.

AI technology is used in a variety of industries, including healthcare, finance, and transportation. For example, in healthcare, AI is used to analyze medical images, such as X-rays and MRIs, to help doctors make more accurate diagnoses. In finance, AI is used to detect fraudulent transactions and prevent financial crimes. In transportation, self-driving cars use AI to
```
