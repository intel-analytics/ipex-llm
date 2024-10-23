# Run Large Multimodal Model on Intel NPU
In this directory, you will find examples on how you could apply IPEX-LLM INT4 or INT8 optimizations on ASR Models on [Intel NPUs](../../../README.md). See the table blow for verified models.

## Verified Models

| Model      | Model Link                                                    |
|------------|----------------------------------------------------------------|
| speech_paraformer-large | [iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch) |

## Requirements
To run these examples with IPEX-LLM on Intel NPUs, make sure to install the newest driver version of Intel NPU.
Go to https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html to download and unzip the driver.
Then go to **Device Manager**, find **Neural Processors** -> **Intel(R) AI Boost**.
Right click and select **Update Driver** -> **Browse my computer for drivers**. And then manually select the unzipped driver folder to install.

## Run Optimized Models
In the example [biciparaformer.py](./biciparaformer.py), we show how to run the paraformer model to transcribe speech to text, with IPEX-LLM low-bit optimizations on Intel NPUs.

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

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the asr repo id for the model (i.e. `iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch`) to be downloaded, or the path to the asr checkpoint folder.
- `--load_in_low_bit`: argument defining the `load_in_low_bit` format used. It is default to be `sym_int8`, `sym_int4` can also be used.

#### Sample Output
##### [iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch)

```log
# speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav
rtf_avg: 0.090: 100%|███████████████████████████████████| 1/1 [00:01<00:00,  1.18s/it]
[{'key': 'asr_example', 'text': '正 是 因 为 存 在 绝 对 正 义 所 以 我 们 接 受 现 实 的 相 对 正 义 但 是 不 要 因 为 现 实 的 相 对 正 义 我 们 就 认 为 这 个 世 界 没 有 正 义 因 为 如 果 当 你 认 为 这 个 世 界 没 有 正 义'}]

# https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav
rtf_avg: 0.232: 100%|███████████████████████████████████| 1/1 [00:01<00:00,  1.29s/it]
[{'key': 'asr_example_zh', 'text': '欢 迎 大 家 来 体 验 达 摩 院 推 出 的 语 音 识 别 模 型'}]
```