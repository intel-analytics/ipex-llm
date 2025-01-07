# Run Large Multimodal Model on Intel NPU
In this directory, you will find examples on how you could apply IPEX-LLM INT4 or INT8 optimizations on Large Multimodal Models on [Intel NPUs](../../../README.md). See the table blow for verified models.

## Verified Models

| Model      | Model Link                                                    |
|------------|----------------------------------------------------------------|
| MiniCPM-Llama3-V-2_5 | [openbmb/MiniCPM-Llama3-V-2_5](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5) |
| MiniCPM-V-2_6 | [openbmb/MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6) |
| Speech_Paraformer-Large | [iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch) |

Please refer to [Quickstart](../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#python-api) for details about verified platforms.

## 0. Prerequisites
For `ipex-llm` NPU support, please refer to [Quickstart](../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#install-prerequisites) for details about the required preparations.

## 1. Install
### 1.1 Installation on Windows
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm

# install ipex-llm with 'npu' option
pip install --pre --upgrade ipex-llm[npu]
pip install torchvision

# [optional] for MiniCPM-V-2_6
pip install timm torch==2.1.2 torchvision==0.16.2

# [optional] for Speech_Paraformer-Large
pip install funasr==1.1.14
pip install modelscope==1.20.1 torch==2.1.2 torchaudio==2.1.2
```
Please refer to [Quickstart](../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#install-ipex-llm-with-npu-support) for more details about `ipex-llm` installation on Intel NPU.

### 1.2 Runtime Configurations
Please refer to [Quickstart](../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#runtime-configurations) for environment variables setting based on your device.

## 2. Run Optimized Models
The examples below show how to run the **_optimized HuggingFace & FunASR model implementations_** on Intel NPU, including
- [MiniCPM-Llama3-V-2_5](./minicpm-llama3-v2.5.py)
- [MiniCPM-V-2_6](./minicpm_v_2_6.py)
- [Speech_Paraformer-Large](./speech_paraformer-large.py)

### 2.1 Run MiniCPM-Llama3-V-2_5 & MiniCPM-V-2_6
```bash
# to run MiniCPM-Llama3-V-2_5
python minicpm-llama3-v2.5.py --save-directory <converted_model_path>

# to run MiniCPM-V-2_6
python minicpm_v_2_6.py --save-directory <converted_model_path>
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the model (e.g. `openbmb/MiniCPM-Llama3-V-2_5` for MiniCPM-Llama3-V-2_5) to be downloaded, or the path to the huggingface checkpoint folder.
- `image-url-or-path IMAGE_URL_OR_PATH`: argument defining the image to be infered. It is default to be 'http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg'.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `"What is in this image?"`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.
- `--max-context-len MAX_CONTEXT_LEN`: argument defining the maximum sequence length for both input and output tokens. It is default to be `1024`.
- `--max-prompt-len MAX_PROMPT_LEN`: argument defining the maximum number of tokens that the input prompt can contain. It is default to be `512`.
- `--low-bit LOW_BIT`: argument defining the low bit optimizations that will be applied to the model. Current available options are `"sym_int4"`, `"asym_int4"` and `"sym_int8"`, with `"sym_int4"` as the default.
- `--save-directory SAVE_DIRECTORY`: argument defining the path to save converted model. If it is a non-existing path, the original pretrained model specified by `REPO_ID_OR_MODEL_PATH` will be loaded, otherwise the lowbit model in `SAVE_DIRECTORY` will be loaded.

#### Troubleshooting

##### Accuracy Tuning
If you enconter output issues when running the examples, you could try the following methods to tune the accuracy:

1. Before running the example, consider setting an additional environment variable `IPEX_LLM_NPU_QUANTIZATION_OPT=1` to enhance output quality.

2. If you are using the default `LOW_BIT` value (i.e. `sym_int4` optimizations), you could try to use `--low-bit "asym_int4"` instead to tune the output quality.

3. You could refer to the [Quickstart](../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#accuracy-tuning) for more accuracy tuning strategies.

> [!IMPORTANT]
> Please note that to make the above methods taking effect, you must specify a new folder for `SAVE_DIRECTORY`. Reusing the same `SAVE_DIRECTORY` will load the previously saved low-bit model, and thus making the above accuracy tuning strategies ineffective.


#### Sample Output
##### [openbmb/MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6)

```log
Inference time: xxxx s
-------------------- Input --------------------
http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg
-------------------- Prompt --------------------
What is in this image?
-------------------- Output --------------------
The image features a young child holding and showing off a white teddy bear wearing a pink dress. The background includes some red flowers and a stone wall, suggesting an outdoor setting.
```

The sample input image is (which is fetched from [COCO dataset](https://cocodataset.org/#explore?id=264959)):

<a href="http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg"><img width=400px src="http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg" ></a>

### 2.2 Run Speech_Paraformer-Large
```bash
# to run Speech_Paraformer-Large
python speech_paraformer-large.py --save-directory <converted_model_path>
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the asr repo id for the model (i.e. `iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch`) to be downloaded, or the path to the asr checkpoint folder.
- `--low-bit LOW_BIT`: argument defining the low bit optimizations that will be applied to the model. It is default to be `sym_int8`, `sym_int4` can also be used.
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
