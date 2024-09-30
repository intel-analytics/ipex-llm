# Run Large Multimodal Model on Intel NPU
In this directory, you will find examples on how you could apply IPEX-LLM INT4 or INT8 optimizations on ASR Models on [Intel NPUs](../../../README.md). See the table blow for verified models.


## 0. Requirements
To run these examples with IPEX-LLM on Intel NPUs, make sure to install the newest driver version of Intel NPU.
Go to https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html to download and unzip the driver.
Then go to **Device Manager**, find **Neural Processors** -> **Intel(R) AI Boost**.
Right click and select **Update Driver** -> **Browse my computer for drivers**. And then manually select the unzipped driver folder to install.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a phi-3-vision model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations on Intel NPUs.
### 1. Install
#### 1.1 Installation on Windows
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.10 libuv
conda activate llm

pip install -U funasr
pip install modelscope torchaudio
# install ipex-llm with 'npu' option
pip install --pre --upgrade ipex-llm[npu]
```

### 2. Runtime Configurations
For optimal performance, it is recommended to set several environment variables. Please check out the suggestions based on your device.
#### 2.1 Configurations for Windows

> [!NOTE]
> For optimal performance, we recommend running code in `conhost` rather than Windows Terminal:
> - Press <kbd>Win</kbd>+<kbd>R</kbd> and input `conhost`, then press Enter to launch `conhost`.
> - Run following command to use conda in `conhost`. Replace `<your conda install location>` with your conda install location.
> ```
> call <your conda install location>\Scripts\activate
> ```

**Following envrionment variables are required**:

```cmd
set BIGDL_USE_NPU=1
```

### 3. Running examples

```
python ./biciparaformer.py
```
