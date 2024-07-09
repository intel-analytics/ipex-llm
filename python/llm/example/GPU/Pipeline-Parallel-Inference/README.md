# Run IPEX-LLM on Multiple Intel GPUs in Pipeline Parallel Fashion

This example demonstrates how to run IPEX-LLM optimized low-bit model vertically partitioned on multiple [Intel GPUs](../README.md) for Linux users.

## Requirements
To run this example with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information. For this particular example, you will need at least two GPUs on your machine.

## Verified Models
- [meta-llama/Llama-2-7b-chat-hf](./run_llama_arc_2_card.sh)
- [meta-llama/Llama-2-13b-chat-hf](./run_llama_arc_2_card.sh)
- [meta-llama/Meta-Llama-3-8B-Instruct](./run_llama_arc_2_card.sh)
- [Qwen/Qwen2-7B-Instruct](./run_qwen2_arc_2_card.sh)
- [Qwen/Qwen1.5-7B-Chat](./run_qwen1.5_arc_2_card.sh)
- [Qwen/Qwen1.5-14B-Chat](./run_qwen1.5_arc_2_card.sh)
- [Qwen/Qwen1.5-32B-Chat](./run_qwen1.5_arc_2_card.sh)
- [Qwen/Qwen1.5-MoE-A2.7B-Chat](./run_qwen1.5_arc_2_card.sh)
- [Qwen/Qwen-VL-Chat](./run_qwen_vl_arc_2_card.sh)
- [Qwen/CodeQwen1.5-7B-Chat](./run_qwen1.5_arc_2_card.sh)
- [THUDM/glm-4-9b-chat](./run_chatglm_arc_2_card.sh)
- [THUDM/glm-4v-9b](./run_glm_4v_arc_2_card.sh)
- [THUDM/chatglm3-6b](./run_chatglm_arc_2_card.sh)
- [baichuan-inc/Baichuan2-7B-Chat](./run_baichuan2_arc_2_card.sh)
- [baichuan-inc/Baichuan2-13B-Chat](./run_baichuan2_arc_2_card.sh)
- [microsoft/Phi-3-mini-4k-instruct](./run_phi3_arc_2_card.sh)
- [microsoft/Phi-3-medium-4k-instruct](./run_phi3_arc_2_card.sh)
- [mistralai/Mistral-7B-v0.1](./run_mistral_arc_2_card.sh)
- [mistralai/Mixtral-8x7B-Instruct-v0.1](./run_mistral_arc_2_card.sh)
- [01-ai/Yi-6B-Chat](./run_yi_arc_2_card.sh)
- [01-ai/Yi-34B-Chat](./run_yi_arc_2_card.sh)
- [codellama/CodeLlama-7b-Instruct-hf](./run_codellama_arc_2_card.sh)
- [codellama/CodeLlama-13b-Instruct-hf](./run_codellama_arc_2_card.sh)
- [codellama/CodeLlama-34b-Instruct-hf](./run_codellama_arc_2_card.sh)
- [upstage/SOLAR-10.7B-Instruct-v1.0](./run_solar_arc_2_card.sh)
- [lmsys/vicuna-7b-v1.3](./run_vicuna_arc_2_card.sh)
- [lmsys/vicuna-13b-v1.3](./run_vicuna_arc_2_card.sh)
- [lmsys/vicuna-33b-v1.3](./run_vicuna_arc_2_card.sh)

## Example: Run pipeline parallel inference on multiple GPUs

### 0. Prerequisites

Please visit the [Install IPEX-LLM on Linux with Intel GPU](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_linux_gpu.html), follow [Install Intel GPU Driver](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_linux_gpu.html#install-intel-gpu-driver) and [Install oneAPI](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_linux_gpu.html#install-oneapi) to install GPU driver and Intel® oneAPI Base Toolkit 2024.0.

### 1. Installation

```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install oneccl_bind_pt==2.1.100 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

### 2. Run pipeline parallel inference on multiple GPUs

For optimal performance, it is recommended to set several environment variables. We provide example usages as following:

> Note: INT4 optimization is applied to the model by default. You could specify other low bit optimizations (such as 'fp8' and 'fp6') through `--low-bit`.

</details>

<details>
  <summary> Show Llama2 and Llama3 example </summary>

#### Run Llama-2-7b-chat-hf / Llama-2-13b-chat-hf / Meta-Llama-3-8B-Instruct on two Intel Arc A770

You could specify `--repo-id-or-model-path` in the test script to be the huggingface repo id for Llama2 / Llama3 to be downloaded, or the path to the huggingface checkpoint folder. Besides, you could change `NUM_GPUS` to the number of GPUs you have on your machine.

```bash
pip install transformers==4.37.0
bash run_llama_arc_2_card.sh
```

</details>

</details>

<details>
  <summary> Show Qwen2 example </summary>

#### Run Qwen2-7B-Instruct on two Intel Arc A770

You could specify `--repo-id-or-model-path` in the test script to be the huggingface repo id for Qwen2 to be downloaded, or the path to the huggingface checkpoint folder. Besides, you could change `NUM_GPUS` to the number of GPUs you have on your machine.

```bash
pip install transformers==4.37.0
bash run_qwen2_arc_2_card.sh
```

</details>

</details>

<details>
  <summary> Show Qwen1.5 example </summary>

#### Run Qwen1.5-7B-Chat / Qwen1.5-14B-Chat / Qwen1.5-32B-Chat / CodeQwen1.5-7B-Chat on two Intel Arc A770

You could specify `--repo-id-or-model-path` in the test script to be the huggingface repo id for Qwen1.5 to be downloaded, or the path to the huggingface checkpoint folder. Besides, you could change `NUM_GPUS` to the number of GPUs you have on your machine.

```bash
pip install transformers==4.37.0
bash run_qwen1.5_arc_2_card.sh
```

#### Run Qwen1.5-MoE-A2.7B-Chat on two Intel Arc A770

You could specify `--repo-id-or-model-path` in the test script to be the huggingface repo id for Qwen1.5-MoE to be downloaded, or the path to the huggingface checkpoint folder. Besides, you could change `NUM_GPUS` to the number of GPUs you have on your machine.

```bash
pip install transformers==4.40.0 trl==0.8.1
bash run_qwen1.5_arc_2_card.sh
```

</details>

</details>

<details>
  <summary> Show Qwen-VL example </summary>

#### Run Qwen-VL-Chat on two Intel Arc A770

You could specify `--repo-id-or-model-path` in the test script to be the huggingface repo id for Qwen-VL to be downloaded, or the path to the huggingface checkpoint folder. Besides, you could change `NUM_GPUS` to the number of GPUs you have on your machine.

```bash
pip install transformers==4.32.0 tiktoken einops transformers_stream_generator==0.0.4 scipy torchvision pillow tensorboard matplotlib
bash run_qwen_vl_arc_2_card.sh
```

</details>

</details>

<details>
  <summary> Show chatglm example </summary>

#### Run glm-4-9b-chat / chatglm3-6B on two Intel Arc A770

You could specify `--repo-id-or-model-path` in the test script to be the huggingface repo id for chatglm to be downloaded, or the path to the huggingface checkpoint folder. Besides, you could change `NUM_GPUS` to the number of GPUs you have on your machine.

```bash
pip install transformers==4.37.0 "tiktoken>=0.7.0"
bash run_chatglm_arc_2_card.sh
```

</details>

<details>
  <summary> Show glm-4v example </summary>

#### Run glm-4v-9b on two Intel Arc A770

You could specify `--repo-id-or-model-path` in the test script to be the huggingface repo id for glm-4v-9b to be downloaded, or the path to the huggingface checkpoint folder. Besides, you could change `NUM_GPUS` to the number of GPUs you have on your machine.

```bash
pip install transformers==4.37.0 tiktoken
bash run_glm_4v_arc_2_card.sh
```

</details>

</details>

<details>
  <summary> Show Baichuan2 example </summary>

#### Run Baichuan2-7B-Chat / Baichuan2-13B-Chat on two Intel Arc A770

You could specify `--repo-id-or-model-path` in the test script to be the huggingface repo id for Baichuan2 to be downloaded, or the path to the huggingface checkpoint folder. Besides, you could change `NUM_GPUS` to the number of GPUs you have on your machine.

```bash
bash run_baichuan2_arc_2_card.sh
```

</details>

</details>

<details>
  <summary> Show Phi3 example </summary>

#### Run Phi-3-mini-4k-instruct / Phi-3-medium-4k-instruct on two Intel Arc A770

You could specify `--repo-id-or-model-path` in the test script to be the huggingface repo id for Phi3 to be downloaded, or the path to the huggingface checkpoint folder. Besides, you could change `NUM_GPUS` to the number of GPUs you have on your machine.

```bash
pip install transformers==4.37.0
bash run_phi3_arc_2_card.sh
```

</details>

</details>

<details>
  <summary> Show Mistral/Mixtral example </summary>

#### Run Mistral-7B-v0.1 / Mixtral-8x7B-Instruct-v0.1 on two Intel Arc A770

You could specify `--repo-id-or-model-path` in the test script to be the huggingface repo id for Mistral / Mixtral to be downloaded, or the path to the huggingface checkpoint folder. Besides, you could change `NUM_GPUS` to the number of GPUs you have on your machine.

```bash
pip install transformers==4.37.0
bash run_mistral_arc_2_card.sh
```

</details>

</details>

<details>
  <summary> Show Yi example </summary>

#### Run Yi-6B-Chat / Yi-34B-Chat on two Intel Arc A770

You could specify `--repo-id-or-model-path` in the test script to be the huggingface repo id for Yi to be downloaded, or the path to the huggingface checkpoint folder. Besides, you could change `NUM_GPUS` to the number of GPUs you have on your machine.

```bash
bash run_yi_arc_2_card.sh
```

</details>

</details>

<details>
  <summary> Show Codellama example </summary>

#### Run CodeLlama-7b-Instruct-hf / CodeLlama-13b-Instruct-hf / CodeLlama-34b-Instruct-hf on two Intel Arc A770

You could specify `--repo-id-or-model-path` in the test script to be the huggingface repo id for Codellama to be downloaded, or the path to the huggingface checkpoint folder. Besides, you could change `NUM_GPUS` to the number of GPUs you have on your machine.

```bash
pip install transformers==4.37.0
bash run_codellama_arc_2_card.sh
```

</details>

</details>

<details>
  <summary> Show Solar example </summary>

#### Run SOLAR-10.7B-Instruct-v1.0 on two Intel Arc A770

You could specify `--repo-id-or-model-path` in the test script to be the huggingface repo id for Solar to be downloaded, or the path to the huggingface checkpoint folder. Besides, you could change `NUM_GPUS` to the number of GPUs you have on your machine.

```bash
pip install transformers==4.37.0
bash run_solar_arc_2_card.sh
```

</details>

</details>

<details>
  <summary> Show Vicuna example </summary>

#### Run vicuna-7b-v1.3 / vicuna-13b-v1.3 / vicuna-33b-v1.3 on two Intel Arc A770

You could specify `--repo-id-or-model-path` in the test script to be the huggingface repo id for Vicuna to be downloaded, or the path to the huggingface checkpoint folder. Besides, you could change `NUM_GPUS` to the number of GPUs you have on your machine.

```bash
bash run_vicuna_arc_2_card.sh
```

</details>


### 3. Sample Output
#### [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
```log
Inference time: xxxx s
First token cost xxxx s and rest tokens cost average xxxx s
-------------------- Prompt --------------------
Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun
-------------------- Output --------------------
Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun. She was always asking her parents to take her on trips, but they were always too busy or too tired.

One day, the little girl
```

#### [Qwen/Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat)
```log
-------------------- Input --------------------
Message: [{'image': 'http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg'}, {'text': '这是什么？'}]
-------------------- Output --------------------
这是一张图片，展现了一个穿着粉色条纹连衣裙的小女孩，她正拿着一只穿粉色裙子的白色玩具小熊。
```
