# Stable Diffusion
In this directory, you will find examples on how to run StableDiffusion models on CPU.

### 1. Installation
#### 1.1 Installation on Linux
We suggest using conda to manage environment. 
```bash
conda create -n diffusion python=3.11
conda activate diffusion
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
pip install diffusers["torch"] transformers
pip install -U PEFT transformers
pip install setuptools==69.5.1
```

#### 1.2 Installation on Windows
We suggest using conda to manage environment. 
```bash
conda create -n diffusion python=3.11 libuv
conda activate diffusion
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
pip install diffusers["torch"] transformers
pip install -U PEFT transformers
pip install setuptools==69.5.1
```

### 2. Examples

#### 2.1 StableDiffusion XL Example
The example shows how to run StableDiffusion XL example on Intel CPU.
```bash
python ./sdxl.py
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the stable diffusion xl model (e.g. `stabilityai/stable-diffusion-xl-base-1.0`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'stabilityai/stable-diffusion-xl-base-1.0'`.
- `--prompt PROMPT`: argument defining the prompt to be infered. It is default to be `'A lovely dog on the table, detailed, 8k'`.
- `--save-path`: argument defining the path to save the generated figure. It is default to be `sdxl-cpu.png`.
- `--num-steps`: argument defining the number of inference steps. It is default to be `20`. 

#### 2.2 LCM-LoRA Example
The example shows how to performing inference with LCM-LoRA on Intel CPU.
```bash
python ./lora-lcm.py
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the stable diffusion xl model (e.g. `stabilityai/stable-diffusion-xl-base-1.0`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'stabilityai/stable-diffusion-xl-base-1.0'`.
- `--lora-weights-path`: argument defining the huggingface repo id for the LCM-LoRA model (e.g. `latent-consistency/lcm-lora-sdxl`) to be downloaded, or the path to huggingface checkpoint folder. It is default to be `'latent-consistency/lcm-lora-sdxl'`. 
- `--prompt PROMPT`: argument defining the prompt to be infered. It is default to be `'A lovely dog on the table, detailed, 8k'`.
- `--save-path`: argument defining the path to save the generated figure. It is default to be `lcm-lora-sdxl-cpu.png`.
- `--num-steps`: argument defining the number of inference steps. It is default to be `4`.
