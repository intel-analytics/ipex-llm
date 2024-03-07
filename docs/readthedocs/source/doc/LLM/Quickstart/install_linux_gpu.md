# Install BigDL-LLM on Linux with Intel GPU

This guide demonstrates how to install BigDL-LLM on Linux with Intel GPUs.

It applies to Intel Data Center GPU Flex Series and Max Series, as well as Intel Arc Series GPU.

Before installation, BigDL-LLM currently supports the Ubuntu 20.04 operating system and later, and supports PyTorch 2.0 and PyTorch 2.1 on Linux. This example installs BigDL-LLM with PyTorch 2.1 using `pip`. For more details like installing with wheel, please refer to [Installation Webpage](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html#linux).


## Install Intel GPU Driver
Install Intel GPU Driver version >= stable_775_20_20231219. We highly recommend installing the latest version of intel-i915-dkms using apt.

  > Note: Please refer to our [driver installation](https://dgpu-docs.intel.com/driver/installation.html) for general purpose GPU capabilities.See [release page](https://dgpu-docs.intel.com/releases/index.html) for latest version.

## Setup Python Environment

* Install the Miniconda as follows
  ```bash
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh
  source ~/.bashrc
  # Verify the installation
  conda --version
  # rm Miniconda3-latest-Linux-x86_64.sh # if you don't need this file any longer
  ```

* Update and install packages
  ```
  conda update -n base conda -y
  conda clean --all --yes
  conda install pip -y
  ```

* Create a new python environment `llm`:
  ```bash
  conda create -n llm python=3.9
  ```

* Activate the newly created environment `llm`:
  ```bash
  conda activate llm
  ```

  <!-- > <img src="https://llm-assets.readthedocs.io/en/latest/_images/quickstart_windows_gpu_5.png"  width=50%; /> -->

## Install oneAPI 

* With the `llm` environment active, install oneAPI in a user-defined folder, e.g. `~/intel/oneapi`.:
  ```bash
  export PYTHONUSERBASE=~/intel/oneapi
  pip install dpcpp-cpp-rt==2024.0.2 mkl-dpcpp==2024.0.0 onednn==2024.0.0 --user
  ```

  > Note: The oneAPI packages are visible in pip list only if PYTHONUSERBASE is properly set.

* Configure your working conda environment (e.g. with name llm) to append oneAPI path (e.g. ~/intel/oneapi/lib) to the environment variable LD_LIBRARY_PATH.
  ```bash
  conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/intel/oneapi/lib -n llm
  ```

## Install `bigdl-llm`

* With the `llm` environment active, use `pip` to install `bigdl-llm` for GPU:
  ```
  conda create -n llm python=3.9
  conda activate llm

  pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
  ```
  > Note: The xpu option will install BigDL-LLM with PyTorch 2.1 by default, which is equivalent to
  ```
  pip install --pre --upgrade bigdl-llm[xpu_2.1] -f https://developer.intel.com/ipex-whl-stable-xpu
  ```

* You can verfy if bigdl-llm is successfully by simply importing a few classes from the library. For example, execute the following import command in terminal:
  ```bash
  python

  > from bigdl.llm.transformers import AutoModel,AutoModelForCausalLM
  ```

## Runtime Configuration

To use GPU acceleration on Linux, several environment variables are required or recommended before running a GPU example.

* For Intel Arc™ A-Series Graphics and Intel Data Center GPU Flex Series, we recommend:
  ```bash
  # Configure oneAPI environment variables. Required step for APT or offline installed oneAPI.
  # Skip this step for PIP-installed oneAPI since the environment has already been configured in LD_LIBRARY_PATH.
  source /opt/intel/oneapi/setvars.sh

  # Recommended Environment Variables for optimal performance
  export USE_XETLA=OFF
  export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
  ```

* For Intel Data Center GPU Max Series, we recommend:
  ```bash
  # Configure oneAPI environment variables. Required step for APT or offline installed oneAPI.
  # Skip this step for PIP-installed oneAPI since the environment has already been configured in LD_LIBRARY_PATH.
  source /opt/intel/oneapi/setvars.sh

  # Recommended Environment Variables for optimal performance
  export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
  export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
  export ENABLE_SDP_FUSION=1
  ```
  Please note that libtcmalloc.so can be installed by conda install -c conda-forge -y gperftools=2.10


## A Quick Example

Now let's play with a real LLM. We'll be using the [phi-1.5](https://huggingface.co/microsoft/phi-1_5) model, a 1.3 billion parameter LLM for this demostration. Follow the steps below to setup and run the model, and observe how it responds to a prompt "What is AI?". 

* Step 1: Open the **Anaconda Prompt** and activate the Python environment `llm` you previously created: 
   ```bash
   conda activate llm
   ```
* Step 2: If you're running on iGPU, set some environment variables by running below commands:
  > For more details about runtime configurations, refer to [this guide](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html#runtime-configuration): 
  ```bash
  # Skip this step for PIP-installed oneAPI since the environment has already been configured in LD_LIBRARY_PATH.
  source /opt/intel/oneapi/setvars.sh

  # Recommended Environment Variables for optimal performance
  export USE_XETLA=OFF
  export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
  ```
* Step 3: To ensure compatibility with `phi-1.5`, update the transformers library to version 4.37.0:
   ```bash
   pip install -U transformers==4.37.0 
   ```
* Step 4: Create a new file named `demo.py` and insert the code snippet below.
   ```python
   # Copy/Paste the contents to a new file demo.py
   import torch
   from bigdl.llm.transformers import AutoModelForCausalLM
   from transformers import AutoTokenizer, GenerationConfig
   generation_config = GenerationConfig(use_cache = True)
   
   tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
   # load Model using bigdl-llm and load it to GPU
   model = AutoModelForCausalLM.from_pretrained(
       "microsoft/phi-1_5", load_in_4bit=True, cpu_embedding=True, trust_remote_code=True)
   model = model.to('xpu')

   # Format the prompt
   question = "What is AI?"
   prompt = " Question:{prompt}\n\n Answer:".format(prompt=question)
   # Generate predicted tokens
   with torch.inference_mode():
       input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu')
       # warm up one more time before the actual generation task for the first run, see details in `Tips & Troubleshooting`
       # output = model.generate(input_ids, do_sample=False, max_new_tokens=32, generation_config = generation_config)
       output = model.generate(input_ids, do_sample=False, max_new_tokens=32, generation_config = generation_config).cpu()
       output_str = tokenizer.decode(output[0], skip_special_tokens=True)
       print(output_str)
   ```
   > Note: when running LLMs on Intel iGPUs with limited memory size, we recommend setting `cpu_embedding=True` in the `from_pretrained` function.
   > This will allow the memory-intensive embedding layer to utilize the CPU instead of GPU.

* Step 5. Run `demo.py` within the activated Python environment using the following command:
  ```bash
  python demo.py
  ```
   
   ### Example output
  
   Example output on a system equipped with an 11th Gen Intel Core i7 CPU and Iris Xe Graphics iGPU:
   ```
   Question:What is AI?
   Answer: AI stands for Artificial Intelligence, which is the simulation of human intelligence in machines.
   ```

## Tips & Troubleshooting

### Warmup for optimial performance on first run
When running LLMs on GPU for the first time, you might notice the performance is lower than expected, with delays up to several minutes before the first token is generated. This delay occurs because the GPU kernels require compilation and initialization, which varies across different GPU models. To achieve optimal and consistent performance, we recommend a one-time warm-up by running `model.generate(...)` an additional time before starting your actual generation tasks. If you're developing an application, you can incorporate this warmup step into start-up or loading routine to enhance the user experience.

