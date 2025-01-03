# Run Large Multimodal Model on Intel NPU
In this directory, you will find examples on how you could apply IPEX-LLM INT4 or INT8 optimizations on Large Multimodal Models on [Intel NPUs](../../../README.md). See the table blow for verified models.

## Verified Models

| Model      | Model Link                                                    |
|------------|----------------------------------------------------------------|
| MiniCPM-Llama3-V-2_5 | [openbmb/MiniCPM-Llama3-V-2_5](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5) |
| MiniCPM-V-2_6 | [openbmb/MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6) |
| Bce-Embedding-Base-V1 | [maidalun1020/bce-embedding-base_v1](https://huggingface.co/maidalun1020/bce-embedding-base_v1) |
| Speech_Paraformer-Large | [iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch) |

Please refer to [Quick Start](../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#python-api) for details about verified platforms.

## 0. Prerequisites
For `ipex-llm` NPU support, please refer to [Quick Start](../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#install-prerequisites) for details about the required preparations.

## 1. Install
### 1.1 Installation on Windows
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.10 libuv
conda activate llm

# install ipex-llm with 'npu' option
pip install --pre --upgrade ipex-llm[npu]
pip install torchvision

# [optional] for MiniCPM-V-2_6
pip install timm torch==2.1.2 torchvision==0.16.2

# [optional] for Bce-Embedding-Base-V1
pip install BCEmbedding==0.1.5 transformers==4.40.0

# [optional] for Speech_Paraformer-Large
pip install funasr==1.1.14
pip install modelscope==1.20.1 torch==2.1.2 torchaudio==2.1.2
```
Please refer to [Quick Start](../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#install-ipex-llm-with-npu-support) for more details about `ipex-llm` installation on Intel NPU.

### 1.2 Runtime Configurations
Please refer to [Quick Start](../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#runtime-configurations) for environment variables setting based on your device.

## 2. Run Optimized Models
The examples below show how to run the **_optimized HuggingFace & FunASR model implementations_** on Intel NPU, including
- [MiniCPM-Llama3-V-2_5](./minicpm-llama3-v2.5.py)
- [MiniCPM-V-2_6](./minicpm_v_2_6.py)
- [Speech_Paraformer-Large](./speech_paraformer-large.py)
- [Bce-Embedding-Base-V1 ](./bce-embedding.py)

### 2.1 Run MiniCPM-Llama3-V-2_5 & MiniCPM-V-2_6
```bash
# to run MiniCPM-Llama3-V-2_5
python minicpm-llama3-v2.5.py --save-directory <converted_model_path>

# to run MiniCPM-V-2_6
python minicpm_v_2_6.py --save-directory <converted_model_path>
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the model (i.e. `openbmb/MiniCPM-Llama3-V-2_5`) to be downloaded, or the path to the huggingface checkpoint folder.
- `image-url-or-path IMAGE_URL_OR_PATH`: argument defining the image to be infered. It is default to be 'http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg'.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `What is in the image?`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.
- `--max-output-len MAX_OUTPUT_LEN`: Defines the maximum sequence length for both input and output tokens. It is default to be `1024`.
- `--max-prompt-len MAX_PROMPT_LEN`: Defines the maximum number of tokens that the input prompt can contain. It is default to be `512`.
- `--disable-transpose-value-cache`: Disable the optimization of transposing value cache.
- `--save-directory SAVE_DIRECTORY`: argument defining the path to save converted model. If it is a non-existing path, the original pretrained model specified by `REPO_ID_OR_MODEL_PATH` will be loaded, otherwise the lowbit model in `SAVE_DIRECTORY` will be loaded.

#### Sample Output
##### [openbmb/MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6)

```log
Inference time: xx.xx s
-------------------- Input --------------------
http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg
-------------------- Prompt --------------------
What is in this image?
-------------------- Output --------------------
The image features a young child holding and showing off a white teddy bear wearing a pink dress. The background includes some red flowers and a stone wall, suggesting an outdoor setting.
```

### 2.2 Run Speech_Paraformer-Large
```bash
# to run Speech_Paraformer-Large
python speech_paraformer-large.py --save-directory <converted_model_path>
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the asr repo id for the model (i.e. `iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch`) to be downloaded, or the path to the asr checkpoint folder.
- `--load_in_low_bit`: argument defining the `load_in_low_bit` format used. It is default to be `sym_int8`, `sym_int4` can also be used.
- `--save-directory SAVE_DIRECTORY`: argument defining the path to save converted model. If it is a non-existing path, the original pretrained model specified by `REPO_ID_OR_MODEL_PATH` will be loaded, otherwise the lowbit model in `SAVE_DIRECTORY` will be loaded.

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

### 2.3 Run Bce-Embedding-Base-V1
```bash
# to run Bce-Embedding-Base-V1
python bce-embedding.py --save-directory <converted_model_path>
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the asr repo id for the model (i.e. `maidalun1020/bce-embedding-base_v1`) to be downloaded, or the path to the asr checkpoint folder.
- `--save-directory SAVE_DIRECTORY`: argument defining the path to save converted model. If it is a non-existing path, the original pretrained model specified by `REPO_ID_OR_MODEL_PATH` will be loaded, otherwise the lowbit model in `SAVE_DIRECTORY` will be loaded.

#### Sample Output
##### [maidalun1020/bce-embedding-base_v1](https://huggingface.co/maidalun1020/bce-embedding-base_v1) |

```log
Inference time: xxx s
[[-0.00674987 -0.01700369 -0.0028928  ... -0.05296675 -0.00352772
   0.00827096]
 [-0.04398304  0.00023038  0.00643183 ... -0.02717186  0.00483789
   0.02298774]]
```
