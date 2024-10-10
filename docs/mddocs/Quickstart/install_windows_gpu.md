# Install IPEX-LLM on Windows with Intel GPU

This guide demonstrates how to install IPEX-LLM on Windows with Intel GPUs. 

It applies to Intel Core Ultra and Core 11 - 14 gen integrated GPUs (iGPUs), as well as Intel Arc Series GPU.

## Table of Contents
- [Install Prerequisites](./install_windows_gpu.md#install-prerequisites)
- [Install ipex-llm](./install_windows_gpu.md#install-ipex-llm)
- [Verify Installation](./install_windows_gpu.md#verify-installation)
- [Monitor GPU Status](./install_windows_gpu.md#monitor-gpu-status)
- [A Quick Example](./install_windows_gpu.md#a-quick-example)
- [Tips & Troubleshooting](./install_windows_gpu.md#tips--troubleshooting)

## Install Prerequisites

### (Optional) Update GPU Driver

> [!IMPORTANT]
> If you have driver version lower than `31.0.101.5122`, it is required to update your GPU driver. Refer to [here](../Overview/install_gpu.md#prerequisites) for more information.

Download and install the latest GPU driver from the [official Intel download page](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html). A system reboot is necessary to apply the changes after the installation is complete.

> [!NOTE]
> The process could take around 10 minutes. After reboot, check for the **Intel Arc Control** application to verify the driver has been installed correctly. If the installation was successful, you should see the **Arc Control** interface similar to the figure below

<img src="https://llm-assets.readthedocs.io/en/latest/_images/quickstart_windows_gpu_3.png" width=100%; />

### Setup Python Environment

Visit [Miniforge installation page](https://conda-forge.org/download/), download the **Miniforge installer for Windows**, and follow the instructions to complete the installation.

<div align="center">
<img src="https://llm-assets.readthedocs.io/en/latest/_images/quickstart_windows_gpu_miniforge_download.png"  width=80%/>
</div>

After installation, open the **Miniforge Prompt**, create a new python environment `llm`:
```cmd
conda create -n llm python=3.11 libuv
```
Activate the newly created environment `llm`:
```cmd
conda activate llm
```
  
## Install `ipex-llm`

With the `llm` environment active, use `pip` to install `ipex-llm` for GPU:

- For **Intel Core™ Ultra Series 2 (a.k.a. Lunar Lake) with Intel Arc™ Graphics**:

   Choose either US or CN website for `extra-index-url`:

   - For **US**:

      ```cmd
      conda create -n llm python=3.11 libuv
      conda activate llm

      pip install --pre --upgrade ipex-llm[xpu_lnl] --extra-index-url --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/lnl/us/
      ```

   - For **CN**:

      ```cmd
      conda create -n llm python=3.11 libuv
      conda activate llm

      pip install --pre --upgrade ipex-llm[xpu_lnl] --extra-index-url --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/lnl/cn/
      ```

- For **other Intel iGPU and dGPU**:

   Choose either US or CN website for `extra-index-url`:

   - For **US**:

      ```cmd
      conda create -n llm python=3.11 libuv
      conda activate llm

      pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
      ```

   - For **CN**:

      ```cmd
      conda create -n llm python=3.11 libuv
      conda activate llm

      pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/
      ```

> [!NOTE]
> If you encounter network issues while installing IPEX, refer to [this guide](../Overview/install_gpu.md#install-ipex-llm-from-wheel) for troubleshooting advice.

## Verify Installation
You can verify if `ipex-llm` is successfully installed following below steps.

### Step 1: Runtime Configurations
- Open the **Miniforge Prompt** and activate the Python environment `llm` you previously created:

   ```cmd
   conda activate llm
   ```

- Set the following environment variables according to your device:

  - For **Intel iGPU**:

    ```cmd
    set SYCL_CACHE_PERSISTENT=1
    set BIGDL_LLM_XMX_DISABLED=1
    ```

  - For **Intel Arc™ A770**:

    ```cmd
    set SYCL_CACHE_PERSISTENT=1
    ```
  
> [!TIP]
> For other Intel dGPU Series, please refer to [this guide](../Overview/install_gpu.md#runtime-configuration) for more details regarding runtime configuration.

### Step 2: Run Python Code

- Launch the Python interactive shell by typing `python` in the Miniforge Prompt window and then press Enter.

- Copy following code to Miniforge Prompt **line by line** and press Enter **after copying each line**.

  ```python
  import torch 
  from ipex_llm.transformers import AutoModel,AutoModelForCausalLM    
  tensor_1 = torch.randn(1, 1, 40, 128).to('xpu') 
  tensor_2 = torch.randn(1, 1, 128, 40).to('xpu') 
  print(torch.matmul(tensor_1, tensor_2).size()) 
  ```

  It will output following content at the end:

  ```
  torch.Size([1, 1, 40, 40])
  ```

  > **Tip**:
  >
  > If you encounter any problem, please refer to [here](../Overview/install_gpu.md#troubleshooting) for help.

- To exit the Python interactive shell, simply press Ctrl+Z then press Enter (or input `exit()` then press Enter).

## Monitor GPU Status
To monitor your GPU's performance and status (e.g. memory consumption, utilization, etc.), you can use either the **Windows Task Manager (in `Performance` Tab)** (see the left side of the figure below) or the **Arc Control** application (see the right side of the figure below)

<img src="https://llm-assets.readthedocs.io/en/latest/_images/quickstart_windows_gpu_4.png"  width=100%; />

## A Quick Example

Now let's play with a real LLM. We'll be using the [Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) model, a 1.8 billion parameter LLM for this demonstration. Follow the steps below to setup and run the model, and observe how it responds to a prompt "What is AI?". 

- Step 1: Follow [Runtime Configurations Section](#step-1-runtime-configurations) above to prepare your runtime environment.

- Step 2: Create code file. IPEX-LLM supports loading model from Hugging Face or ModelScope. Please choose according to your requirements.

  - For **loading model from Hugging Face**:
    
    Create a new file named `demo.py` and insert the code snippet below to run [Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) model with IPEX-LLM optimizations.

      ```python
      # Copy/Paste the contents to a new file demo.py
      import torch
      from ipex_llm.transformers import AutoModelForCausalLM
      from transformers import AutoTokenizer, GenerationConfig
      generation_config = GenerationConfig(use_cache=True)

      print('Now start loading Tokenizer and optimizing Model...')
      tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct",
                                                trust_remote_code=True)

      # Load Model using ipex-llm and load it to GPU
      model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct",
                                                   load_in_4bit=True,
                                                   cpu_embedding=True,
                                                   trust_remote_code=True)
      model = model.to('xpu')
      print('Successfully loaded Tokenizer and optimized Model!')

      # Format the prompt
      # you could tune the prompt based on your own model,
      # here the prompt tuning refers to https://huggingface.co/Qwen/Qwen2-1.5B-Instruct#quickstart
      question = "What is AI?"
      messages = [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": question}
      ]
      text = tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
      )

      # Generate predicted tokens
      with torch.inference_mode():
         input_ids = tokenizer.encode(text, return_tensors="pt").to('xpu')

         print('--------------------------------------Note-----------------------------------------')
         print('| For the first time that each model runs on Intel iGPU/Intel Arc™ A300-Series or |')
         print('| Pro A60, it may take several minutes for GPU kernels to compile and initialize. |')
         print('| Please be patient until it finishes warm-up...                                  |')
         print('-----------------------------------------------------------------------------------')

         # To achieve optimal and consistent performance, we recommend a one-time warm-up by running `model.generate(...)` an additional time before starting your actual generation tasks.
         # If you're developing an application, you can incorporate this warm-up step into start-up or loading routine to enhance the user experience.
         output = model.generate(input_ids,
                                 do_sample=False,
                                 max_new_tokens=32,
                                 generation_config=generation_config) # warm-up

         print('Successfully finished warm-up, now start generation...')

         output = model.generate(input_ids,
                                 do_sample=False,
                                 max_new_tokens=32,
                                 generation_config=generation_config).cpu()
         output_str = tokenizer.decode(output[0], skip_special_tokens=False)
         print(output_str)
      ```
  - For **loading model ModelScopee**:

    Please first run following command in Miniforge Prompt to install ModelScope:
    ```cmd
    pip install modelscope==1.11.0
    ```

    Create a new file named `demo.py` and insert the code snippet below to run [Qwen2-1.5B-Instruct](https://www.modelscope.cn/models/qwen/Qwen2-1.5B-Instruct/summary) model with IPEX-LLM optimizations.

      ```python
      # Copy/Paste the contents to a new file demo.py
      import torch
      from ipex_llm.transformers import AutoModelForCausalLM
      from transformers import GenerationConfig
      from modelscope import AutoTokenizer
      generation_config = GenerationConfig(use_cache=True)

      print('Now start loading Tokenizer and optimizing Model...')
      tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct",
                                                trust_remote_code=True)

      # Load Model using ipex-llm and load it to GPU
      model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct",
                                                   load_in_4bit=True,
                                                   cpu_embedding=True,
                                                   trust_remote_code=True,
                                                   model_hub='modelscope')
      model = model.to('xpu')
      print('Successfully loaded Tokenizer and optimized Model!')

      # Format the prompt
      # you could tune the prompt based on your own model,
      # here the prompt tuning refers to https://huggingface.co/Qwen/Qwen2-1.5B-Instruct#quickstart
      question = "What is AI?"
      messages = [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": question}
      ]
      text = tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
      )
      
      # Generate predicted tokens
      with torch.inference_mode():
         input_ids = tokenizer.encode(text, return_tensors="pt").to('xpu')
         print('--------------------------------------Note-----------------------------------------')
         print('| For the first time that each model runs on Intel iGPU/Intel Arc™ A300-Series or |')
         print('| Pro A60, it may take several minutes for GPU kernels to compile and initialize. |')
         print('| Please be patient until it finishes warm-up...                                  |')
         print('-----------------------------------------------------------------------------------')

         # To achieve optimal and consistent performance, we recommend a one-time warm-up by running `model.generate(...)` an additional time before starting your actual generation tasks.
         # If you're developing an application, you can incorporate this warm-up step into start-up or loading routine to enhance the user experience.
         output = model.generate(input_ids,
                                 do_sample=False,
                                 max_new_tokens=32,
                                 generation_config=generation_config) # warm-up

         print('Successfully finished warm-up, now start generation...')

         output = model.generate(input_ids,
                                 do_sample=False,
                                 max_new_tokens=32,
                                 generation_config=generation_config).cpu()
         output_str = tokenizer.decode(output[0], skip_special_tokens=False)
         print(output_str)
      ```
      > **Note**:
      >
      > Please note that the repo id on ModelScope may be different from Hugging Face for some models.

> [!NOTE]
> When running LLMs on Intel iGPUs with limited memory size, we recommend setting `cpu_embedding=True` in the `from_pretrained` function.
> This will allow the memory-intensive embedding layer to utilize the CPU instead of GPU.

- Step 3. Run `demo.py` within the activated Python environment using the following command:

  ```cmd
  python demo.py
  ```
   
### Example output

Example output on a system equipped with an Intel Core Ultra 5 125H CPU and Intel Arc Graphics iGPU:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is AI?<|im_end|>
<|im_start|>assistant
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and act like humans. It involves the development of algorithms,
```

## Tips & Troubleshooting

### Warm-up for optimal performance on first run
When running LLMs on GPU for the first time, you might notice the performance is lower than expected, with delays up to several minutes before the first token is generated. This delay occurs because the GPU kernels require compilation and initialization, which varies across different GPU types. To achieve optimal and consistent performance, we recommend a one-time warm-up by running `model.generate(...)` an additional time before starting your actual generation tasks. If you're developing an application, you can incorporate this warm-up step into start-up or loading routine to enhance the user experience.
