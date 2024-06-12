# Stable Diffusion
In this directory, you will find examples on how to run StableDiffusion models on CPU.

### 1. Installation
#### 1.1. Install IPEX-LLM
Follow the instructions in [IPEX-LLM CPU installation guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Overview/install_cpu.html) to install ipex-llm. We recommend to use miniconda to manage your python environment.

#### 1.2 Install dependencies for Stable Diffusion
Assume you have created a conda environment named diffusion with ipex-llm installed. Run below commands to install dependencies for running Stable Diffusion.
```bash
conda activate diffusion
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
- `--prompt PROMPT`: argument defining the prompt to be infered. It is default to be `'An astronaut in the forest, detailed, 8k'`.
- `--save-path`: argument defining the path to save the generated figure. It is default to be `sdxl-cpu.png`.
- `--num-steps`: argument defining the number of inference steps. It is default to be `20`. 

The sample output image looks like below. 
![image](https://llm-assets.readthedocs.io/en/latest/_images/sdxl-cpu.png)

#### 4.2 LCM-LoRA Example
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