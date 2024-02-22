# Install BigDL-LLM on Windows for Intel GPU

## MTL & iGPU & Arc

### Install GPU driver

* Install Visual Studio 2022 Community Edition from [here](https://visualstudio.microsoft.com/downloads/). 

  > Note select `Desktop development with C++` during installation.
  > The installation could be slow and cost 15 minutes. Need at least 7GB. 
  > If you do not select this workload during installation, go to Tools > Get Tools and Features... to change workload following [this page](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170#step-4---choose-workloads). 
  > <img src="https://llm-assets.readthedocs.io/en/latest/_images/quickstart_windows_gpu_1.png" alt="image-20240221102252560" width=80%; />

* Install latest GPU driver from [here](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html). Note the process could be slow and It takes 10 minutes to download and install. Reboot is also needed. 
After rebooting, if driver is installed correctly we will see the Arc Control like the fig below. 
  >  <img src="https://llm-assets.readthedocs.io/en/latest/_images/quickstart_windows_gpu_3.png" width=70%; />
  > 
  > We can check GPU status from Arc Control (the left one in fig) or Task Manager (the right one in fig). 
  >  <img src="https://llm-assets.readthedocs.io/en/latest/_images/quickstart_windows_gpu_4.png"  width=70%; />

### Install conda

We recommend using miniconda to create environment. Please refer to the [page](https://docs.anaconda.com/free/miniconda/) to install miniconda. 

* Choose windows miniconda installer. Download and install. It takes a few minutes. 

  > <img src="https://llm-assets.readthedocs.io/en/latest/_images/quickstart_windows_gpu_5.png"  width=50%; />

* After installation, open `Anaconda prompt` and create an environment by `conda create -n llm python=3.9 libuv` . 

  > Note: if you encounter CondaHTTPError problem and fail to create the environment, please check the internet connection and proxy setting. You can define your proxy setting by `conda config --set proxy_servers.http your_http_proxy_IP:port` and `conda config --set proxy_servers.https your_https_proxy_IP:port`
  >
  >  <img src="https://llm-assets.readthedocs.io/en/latest/_images/quickstart_windows_gpu_6.png"  width=50%; />

### Install oneAPI 

* Install oneAPI Base Toolkit with the help of pip. After ensuring  `conda` is ready, we can use `pip ` to install oneAPI Base Toolkit. 

  ```bash
  pip install dpcpp-cpp-rt==2024.0.2 mkl-dpcpp==2024.0.0 onednn==2024.0.0
  ```

  > If you encounter HTTP Timeout error, also check your internet and proxy setting in `pip.ini` file which is under "C:\Users\YourName\AppData\Roaming\pip"  folder. 

### Install bigdl-llm

* Run the commands below in Anaconda prompt. 

  ```bash
  conda activate llm

  pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
  ```


* Now you can test whether all the components have been installed correctly within the interactive python prompt. If we can import all the packages correctly following the python file below, then the installation is correct. 
  ```python
  import torch
  import time
  import argparse
  import numpy as np
    
  from bigdl.llm.transformers import AutoModel,AutoModelForCausalLM
  from transformers import AutoTokenizer, GenerationConfig
  ```

### A quick example
Then we use phi-1.5 as an example to show how to run the model with bigdl-llm on windows. Here we we provide `demo.py` and you can run it with `python demo.py`. 
> Note that transformer version should match the model you want to use. For example, here we use transformers 4.37.0 to run the demo. 
> ```
> pip install transformers==4.37.0 
> ```

```python
  # demo.py
  import torch
  import numpy as np
  from bigdl.llm.transformers import AutoModelForCausalLM
  from transformers import AutoTokenizer, GenerationConfig

  PHI1_5_PROMPT_FORMAT = " Question:{prompt}\n\n Answer:"
  generation_config = GenerationConfig(use_cache = True)

  if __name__ == '__main__':
      model_path = "microsoft/phi-1_5"
      prompt = "What is AI?"
      n_predict = 32
      # Load model in 4 bit,
      # which convert the relevant layers in the model into INT4 format
      # When running LLMs on Intel iGPUs for Windows users, we recommend setting `cpu_embedding=True` in the from_pretrained function.
      # This will allow the memory-intensive embedding layer to utilize the CPU instead of iGPU.
      model = AutoModelForCausalLM.from_pretrained(model_path,
                                                  load_in_4bit=True,
                                                  # cpu_embedding=True,
                                                  trust_remote_code=True)

      model = model.to('xpu')

      # Load tokenizer
      tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                trust_remote_code=True)
      
      # Generate predicted tokens
      with torch.inference_mode():
          prompt = PHI1_5_PROMPT_FORMAT.format(prompt=prompt)
          input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu')
          output = model.generate(input_ids, do_sample=False, max_new_tokens=n_predict, generation_config = generation_config)
          torch.xpu.synchronize()
          output = output.cpu()
          output_str = tokenizer.decode(output[0], skip_special_tokens=True)
          print('-'*20, 'Prompt', '-'*20)
          print(prompt)
          print('-'*20, 'Output', '-'*20)
          print(output_str)
   ```
   Here is the sample output on the laptop equipped with 11th Gen Intel(R) Core(TM) i7-1185G7 and Intel(R) Iris(R) Xe Graphics after running the example program above. 
   ```
   Inference time: 3.526491641998291 s
   -------------------- Prompt --------------------
   Question:What is AI?

   Answer:
   -------------------- Output --------------------
   Question:What is AI?

   Answer: AI stands for Artificial Intelligence, which is the simulation of human intelligence in machines.
   ```

   

