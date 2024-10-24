# Install IPEX-LLM on Linux with Intel GPU

This guide demonstrates how to install IPEX-LLM on Linux with Intel GPUs. It applies to Intel Data Center GPU Flex Series and Max Series, as well as Intel Arc Series GPU and Intel iGPU.

IPEX-LLM currently supports the Ubuntu 20.04 operating system and later, and supports PyTorch 2.0 and PyTorch 2.1 on Linux. This page demonstrates IPEX-LLM with PyTorch 2.1. Check the [Installation](../Overview/install_gpu.md#linux) page for more details.


## Table of Contents
- [Install Prerequisites](./install_linux_gpu.md#install-prerequisites)
  - [For Intel Core™ Ultra Processors with Intel Arc™ Graphics (a.k.a. Meteor Lake)](#for-intel-core-ultra-processors-with-intel-arc-graphics-aka-meteor-lake)
  - [For other Intel iGPU and dGPU](#for-other-intel-igpu-and-dgpu)
- [Install ipex-llm](./install_linux_gpu.md#install-ipex-llm)
- [Verify Installation](./install_linux_gpu.md#verify-installation)
- [Runtime Configurations](./install_linux_gpu.md#runtime-configurations)
- [A Quick Example](./install_linux_gpu.md#a-quick-example)
- [Tips & Troubleshooting](./install_linux_gpu.md#tips--troubleshooting)

## Install Prerequisites

### For Intel Core™ Ultra Processors with Intel Arc™ Graphics (a.k.a. Meteor Lake)

> [!NOTE]
> For IPEX-LLM on Linux with Meteor Lake integrated GPU, we have currently verified on Ubuntu 22.04 with kernel `6.5.0-35-generic`.

#### 1. Check current kernel version

You could check your current kernel version through:

```bash
uname -r
```
If the version displayed is not `6.5.0-35-generic`, downgrade or upgrade the kernel to the recommended version.

#### 2. (Optional) Downgrade / Upgrade to kernel 6.5.0-35

If your current kernel version is not `6.5.0-35-generic`, you could downgrade or upgrade it by:

```bash
export VERSION="6.5.0-35"
sudo apt-get install -y linux-headers-$VERSION-generic
sudo apt-get install -y linux-image-$VERSION-generic
sudo apt-get install -y linux-modules-extra-$VERSION-generic

sudo sed -i "s/GRUB_DEFAULT=.*/GRUB_DEFAULT=\"1> $(echo $(($(awk -F\' '/menuentry / {print $2}' /boot/grub/grub.cfg \
| grep -no $VERSION | sed 's/:/\n/g' | head -n 1)-2)))\"/" /etc/default/grub

sudo update-grub
```

And the reboot your machine:

```bash
sudo reboot
```

After rebooting, you can use `uname -r` again to see that your kernel version has been changed to `6.5.0-35-generic`.

#### 3. Enable driver support through `force_probe` flag

Next, you need to enable driver support on kernel `6.5.0-35-generic` through `force_probe` parameter：

```bash
sudo sed -i "s/GRUB_CMDLINE_LINUX_DEFAULT=\"quiet splash\"/GRUB_CMDLINE_LINUX_DEFAULT=\"quiet splash i915.force_probe=7d55\"/g" /etc/default/grub
```

> [!TIP]
> In addition to using the `sed` command, you could also modify the `/etc/default/grub` file directly to make sure `GRUB_CMDLINE_LINUX_DEFAULT="quiet splash i915.force_probe=7d55"`

Reboot the machine then to make the configuration take effect:

```bash
sudo reboot
```

#### 4. Install compute packages

Compute packages are also required to be installed for Intel GPU on Ubuntu 22.04 through the following commands:

```bash
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg

echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list

sudo apt update

apt-get install -y libze1 intel-level-zero-gpu intel-opencl-icd clinfo
```


#### 5. Configure permmision and verify driver setup

To complete the driver setup, you need to make sure your user is in the render group:

```bash
sudo gpasswd -a ${USER} render
newgrp render
```

You could then verify whether the driver is functioning properly with:

```bash
clinfo | grep "Device Name"
```

whose output should contain `Intel(R) Arc(TM) Graphics`.

> [!TIP]
> You could refer to the [official driver guide for client GPUS](https://dgpu-docs.intel.com/driver/client/overview.html#installing-client-gpus-on-ubuntu-desktop-22-04-lts) for more information.

#### 6. Install OneAPI

The final step for prerequisites installation on Meteor Lake iGPU is to install OneAPI 2024.0:

```bash
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list

sudo apt update

sudo apt install intel-oneapi-common-vars=2024.0.0-49406 \
  intel-oneapi-common-oneapi-vars=2024.0.0-49406 \
  intel-oneapi-diagnostics-utility=2024.0.0-49093 \
  intel-oneapi-compiler-dpcpp-cpp=2024.0.2-49895 \
  intel-oneapi-dpcpp-ct=2024.0.0-49381 \
  intel-oneapi-mkl=2024.0.0-49656 \
  intel-oneapi-mkl-devel=2024.0.0-49656 \
  intel-oneapi-mpi=2021.11.0-49493 \
  intel-oneapi-mpi-devel=2021.11.0-49493 \
  intel-oneapi-dal=2024.0.1-25 \
  intel-oneapi-dal-devel=2024.0.1-25 \
  intel-oneapi-ippcp=2021.9.1-5 \
  intel-oneapi-ippcp-devel=2021.9.1-5 \
  intel-oneapi-ipp=2021.10.1-13 \
  intel-oneapi-ipp-devel=2021.10.1-13 \
  intel-oneapi-tlt=2024.0.0-352 \
  intel-oneapi-ccl=2021.11.2-5 \
  intel-oneapi-ccl-devel=2021.11.2-5 \
  intel-oneapi-dnnl-devel=2024.0.0-49521 \
  intel-oneapi-dnnl=2024.0.0-49521 \
  intel-oneapi-tcm-1.0=1.0.0-435
```

>[!IMPORTANT]
> Please make sure to reboot the machine after all the prerequiste steps are complete:
>
> ```bash
> sudo reboot
> ```

### For other Intel iGPU and dGPU

#### 1. Install GPU Driver

##### For Linux kernel 6.2

* Choose one option below depending on your CPU type:

  1. **Option 1**: For `Intel Core CPU` with multiple A770 Arc GPUs. Use the following repository:
      ```bash
      sudo apt-get install -y gpg-agent wget
      wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
      sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
      echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" | \
      sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list
      ```

  2. **Option 2**: For `Intel Xeon-W/SP CPU` with multiple A770 Arc GPUs. Use this repository for better performance:
      ```bash
      wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
      sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
      echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy/lts/2350 unified" | \
      sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list
      sudo apt update
      ```

    <img src="https://llm-assets.readthedocs.io/en/latest/_images/wget.png" width=100%; />

* Install drivers

    ```bash
    sudo apt-get update

    # Install out-of-tree driver
    sudo apt-get -y install \
        gawk \
        dkms \
        linux-headers-$(uname -r) \
        libc6-dev
    sudo apt install intel-i915-dkms intel-fw-gpu

    # Install Compute Runtime
    sudo apt-get install -y udev \
        intel-opencl-icd intel-level-zero-gpu level-zero \
        intel-media-va-driver-non-free libmfx1 libmfxgen1 libvpl2 \
        libegl-mesa0 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
        libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers \
        mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all vainfo
    
    sudo reboot
    ```

    <img src="https://llm-assets.readthedocs.io/en/latest/_images/i915.png" width=100%; />

    <img src="https://llm-assets.readthedocs.io/en/latest/_images/gawk.png" width=100%; />


* Configure permissions
    ```bash
    sudo gpasswd -a ${USER} render
    newgrp render

    # Verify the device is working with i915 driver
    sudo apt-get install -y hwinfo
    hwinfo --display
    ```

##### For Linux kernel 6.5

* Choose one option below depending on your CPU type:

  1. **Option 1**: For `Intel Core CPU` with multiple A770 Arc GPUs. Use the following repository:
      ```bash
      sudo apt-get install -y gpg-agent wget
      wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
      sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
      echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" | \
      sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list
      ```

  2. **Option 2**: For `Intel Xeon-W/SP CPU` with multiple A770 Arc GPUs. Use this repository for better performance:
      ```bash
      wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
      sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
      echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy/lts/2350 unified" | \
      sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list
      sudo apt update
      ```

    <img src="https://llm-assets.readthedocs.io/en/latest/_images/wget.png" width=100%; />

* Install drivers

    ```bash
    sudo apt-get update

    # Install out-of-tree driver
    sudo apt-get -y install \
        gawk \
        dkms \
        linux-headers-$(uname -r) \
        libc6-dev
    sudo apt install -y intel-i915-dkms intel-fw-gpu

    # Install Compute Runtime
    sudo apt-get install -y udev \
        intel-opencl-icd intel-level-zero-gpu level-zero \
        intel-media-va-driver-non-free libmfx1 libmfxgen1 libvpl2 \
        libegl-mesa0 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
        libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers \
        mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all vainfo
  
    sudo reboot
    ```

    <img src="https://llm-assets.readthedocs.io/en/latest/_images/gawk.png" width=100%; />

* Configure permissions
    ```bash
    sudo gpasswd -a ${USER} render
    newgrp render

    # Verify the device is working with i915 driver
    sudo apt-get install -y hwinfo
    hwinfo --display
    ```

<!-- #### (Optional) Update Level Zero on Intel Core™ Ultra iGPU
For Intel Core™ Ultra integrated GPU, please make sure level_zero version >= 1.3.28717. The level_zero version can be checked with `sycl-ls`, and verison will be tagged behind `[ext_oneapi_level_zero:gpu]`.

Here are the sample output of `sycl-ls`:
```bash
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device OpenCL 1.2  [2023.16.12.0.12_195853.xmain-hotfix]
[opencl:cpu:1] Intel(R) OpenCL, Intel(R) Core(TM) Ultra 5 125H OpenCL 3.0 (Build 0) [2023.16.12.0.12_195853.xmain-hotfix]
[opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) Graphics OpenCL 3.0 NEO  [24.09.28717.12]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Arc(TM) Graphics 1.3 [1.3.28717]
```

If you have level_zero version < 1.3.28717, you could update as follows:
```bash
wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.16238.4/intel-igc-core_1.0.16238.4_amd64.deb
wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.16238.4/intel-igc-opencl_1.0.16238.4_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/24.09.28717.12/intel-level-zero-gpu-dbgsym_1.3.28717.12_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/24.09.28717.12/intel-level-zero-gpu_1.3.28717.12_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/24.09.28717.12/intel-opencl-icd-dbgsym_24.09.28717.12_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/24.09.28717.12/intel-opencl-icd_24.09.28717.12_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/24.09.28717.12/libigdgmm12_22.3.17_amd64.deb
sudo dpkg -i *.deb
``` -->

#### 2. Install oneAPI 
  ```bash
  wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

  echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
  
  sudo apt update

  sudo apt install intel-oneapi-common-vars=2024.0.0-49406 \
    intel-oneapi-common-oneapi-vars=2024.0.0-49406 \
    intel-oneapi-diagnostics-utility=2024.0.0-49093 \
    intel-oneapi-compiler-dpcpp-cpp=2024.0.2-49895 \
    intel-oneapi-dpcpp-ct=2024.0.0-49381 \
    intel-oneapi-mkl=2024.0.0-49656 \
    intel-oneapi-mkl-devel=2024.0.0-49656 \
    intel-oneapi-mpi=2021.11.0-49493 \
    intel-oneapi-mpi-devel=2021.11.0-49493 \
    intel-oneapi-dal=2024.0.1-25 \
    intel-oneapi-dal-devel=2024.0.1-25 \
    intel-oneapi-ippcp=2021.9.1-5 \
    intel-oneapi-ippcp-devel=2021.9.1-5 \
    intel-oneapi-ipp=2021.10.1-13 \
    intel-oneapi-ipp-devel=2021.10.1-13 \
    intel-oneapi-tlt=2024.0.0-352 \
    intel-oneapi-ccl=2021.11.2-5 \
    intel-oneapi-ccl-devel=2021.11.2-5 \
    intel-oneapi-dnnl-devel=2024.0.0-49521 \
    intel-oneapi-dnnl=2024.0.0-49521 \
    intel-oneapi-tcm-1.0=1.0.0-435
  ```
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/oneapi.png" alt="image-20240221102252565" width=100%; />

  <img src="https://llm-assets.readthedocs.io/en/latest/_images/basekit.png" alt="image-20240221102252565" width=100%; />

### Setup Python Environment
 
Download and install the Miniforge as follows if you don't have conda installed on your machine:
  ```bash
  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash Miniforge3-Linux-x86_64.sh
  source ~/.bashrc
  ```

You can use `conda --version` to verify you conda installation.

After installation, create a new python environment `llm`:
```bash
conda create -n llm python=3.11
```
Activate the newly created environment `llm`:
```bash
conda activate llm
```


## Install `ipex-llm`

With the `llm` environment active, use `pip` to install `ipex-llm` for GPU. Choose either US or CN website for `extra-index-url`:

- For **US**:

  ```bash
  pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
  ```

- For **CN**:

  ```bash
  pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/
  ```

> [!NOTE]
> If you encounter network issues while installing IPEX, refer to [this guide](../Overview/install_gpu.md#install-ipex-llm-from-wheel-1) for troubleshooting advice.


## Verify Installation
- You can verify if `ipex-llm` is successfully installed by simply importing a few classes from the library. For example, execute the following import command in the terminal:

  ```bash
  source /opt/intel/oneapi/setvars.sh

  python

  > from ipex_llm.transformers import AutoModel, AutoModelForCausalLM
  ```

## Runtime Configurations

To use GPU acceleration on Linux, several environment variables are required or recommended before running a GPU example. Choose corresponding configurations based on your GPU device:

- For **Intel Arc™ A-Series and Intel Data Center GPU Flex**:

  For Intel Arc™ A-Series Graphics and Intel Data Center GPU Flex Series, we recommend:

  ```bash
  # Configure oneAPI environment variables.
  source /opt/intel/oneapi/setvars.sh

  # Recommended Environment Variables for optimal performance
  export USE_XETLA=OFF
  export SYCL_CACHE_PERSISTENT=1
  # [optional] under most circumstances, the following environment variable may improve performance, but sometimes this may also cause performance degradation
  export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
  ```

- For **Intel Data Center GPU Max**:

  For Intel Data Center GPU Max Series, we recommend:

  ```bash
  # Configure oneAPI environment variables.
  source /opt/intel/oneapi/setvars.sh

  # Recommended Environment Variables for optimal performance
  export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
  export SYCL_CACHE_PERSISTENT=1
  export ENABLE_SDP_FUSION=1
  # [optional] under most circumstances, the following environment variable may improve performance, but sometimes this may also cause performance degradation
  export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
  ```

  Please note that `libtcmalloc.so` can be installed by `conda install -c conda-forge -y gperftools=2.10`

  - For **Intel iGPU**:

    ```bash
    # Configure oneAPI environment variables. 
    source /opt/intel/oneapi/setvars.sh

    export SYCL_CACHE_PERSISTENT=1
    export BIGDL_LLM_XMX_DISABLED=1
    ```

> [!NOTE]
> Please refer to [this guide](../Overview/install_gpu.md#runtime-configuration-1) for more details regarding runtime configuration.

> [!NOTE]
> The environment variable `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` determines the usage of immediate command lists for task submission to the GPU. While this mode typically enhances performance, exceptions may occur. Please consider experimenting with and without this environment variable for best performance. For more details, you can refer to [this article](https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html).

## A Quick Example

Now let's play with a real LLM. We'll be using the [phi-1.5](https://huggingface.co/microsoft/phi-1_5) model, a 1.3 billion parameter LLM for this demostration. Follow the steps below to setup and run the model, and observe how it responds to a prompt "What is AI?". 

- Step 1: Activate the Python environment `llm` you previously created:

   ```bash
   conda activate llm
   ```

- Step 2: Follow [Runtime Configurations Section](#runtime-configurations) above to prepare your runtime environment.

- Step 3: Create a new file named `demo.py` and insert the code snippet below.

   ```python
   # Copy/Paste the contents to a new file demo.py
   import torch
   from ipex_llm.transformers import AutoModelForCausalLM
   from transformers import AutoTokenizer, GenerationConfig
   generation_config = GenerationConfig(use_cache = True)
   
   tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b", trust_remote_code=True)
   # load Model using ipex-llm and load it to GPU
   model = AutoModelForCausalLM.from_pretrained(
       "tiiuae/falcon-7b", load_in_4bit=True, cpu_embedding=True, trust_remote_code=True)
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

   > **Note**:
   >
   > When running LLMs on Intel iGPUs with limited memory size, we recommend setting `cpu_embedding=True` in the `from_pretrained` function.
   > This will allow the memory-intensive embedding layer to utilize the CPU instead of GPU.

- Step 5. Run `demo.py` within the activated Python environment using the following command:

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
When running LLMs on GPU for the first time, you might notice the performance is lower than expected, with delays up to several minutes before the first token is generated. This delay occurs because the GPU kernels require compilation and initialization, which varies across different GPU types. To achieve optimal and consistent performance, we recommend a one-time warm-up by running `model.generate(...)` an additional time before starting your actual generation tasks. If you're developing an application, you can incorporate this warmup step into start-up or loading routine to enhance the user experience.

