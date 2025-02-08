# C++ Example of running LLM on Intel NPU using IPEX-LLM
In this directory, you will find a C++ example on how to run LLM models on Intel NPUs using IPEX-LLM (leveraging *Intel NPU Acceleration Library*). See the table blow for verified models.

## Verified Models

| Model      | Model Link                                                    |
|------------|----------------------------------------------------------------|
| Llama2 | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| Llama3 | [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| Llama3.2 | [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| Qwen2 | [Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct), [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) |
| Qwen2.5 | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| MiniCPM | [openbmb/MiniCPM-1B-sft-bf16](https://huggingface.co/openbmb/MiniCPM-1B-sft-bf16), [openbmb/MiniCPM-2B-sft-bf16](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16) |
| DeepSeek-R1 | [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B), [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) |

Please refer to [Quickstart](../../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#c-api) for details about verified platforms.

## 0. Prerequisites
For `ipex-llm` NPU support, please refer to [Quickstart](../../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#install-prerequisites) for details about the required preparations.

## 1. Install & Runtime Configurations
### 1.1 Installation on Windows
We suggest using conda to manage environment:
```cmd
conda create -n llm python=3.11
conda activate llm

:: for building the example
pip install cmake

:: install ipex-llm with 'npu' option
pip install --pre --upgrade ipex-llm[npu]

:: [optional] for Llama-3.2-1B-Instruct & Llama-3.2-3B-Instruct
pip install transformers==4.45.0 accelerate==0.33.0
```

Please refer to [Quickstart](../../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#install-prerequisites) for more details about `ipex-llm` installation on Intel NPU.

### 1.2 Runtime Configurations
Please refer to [Quickstart](../../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#runtime-configurations) for environment variables setting based on your device.

## 2. Convert Model
We provide a [convert script](convert.py) under current directory, by running it, you can obtain the whole weights and configuration files which are required to run C++ example.

```cmd
:: to convert Llama-2-7b-chat-hf
python convert.py --repo-id-or-model-path meta-llama/Llama-2-7b-chat-hf --save-directory <converted_model_path>

:: to convert Meta-Llama-3-8B-Instruct
python convert.py --repo-id-or-model-path meta-llama/Meta-Llama-3-8B-Instruct --save-directory <converted_model_path>

:: to convert Llama-3.2-1B-Instruct
python convert.py --repo-id-or-model-path meta-llama/Llama-3.2-1B-Instruct --save-directory <converted_model_path>

:: to convert Llama-3.2-3B-Instruct
python convert.py --repo-id-or-model-path meta-llama/Llama-3.2-3B-Instruct --save-directory <converted_model_path>

:: to convert Qwen2-1.5B-Instruct
python convert.py --repo-id-or-model-path Qwen/Qwen2-1.5B-Instruct --save-directory <converted_model_path>

:: to convert Qwen2-7B-Instruct
python convert.py --repo-id-or-model-path Qwen/Qwen2-7B-Instruct --save-directory <converted_model_path>

:: to convert Qwen2.5-3B-Instruct
python convert.py --repo-id-or-model-path Qwen/Qwen2.5-3B-Instruct --save-directory <converted_model_path> --low-bit "asym_int4"

:: to convert Qwen2.5-7B-Instruct
python convert.py --repo-id-or-model-path Qwen/Qwen2.5-7B-Instruct --save-directory <converted_model_path>

:: to convert MiniCPM-1B-sft-bf16
python convert.py --repo-id-or-model-path openbmb/MiniCPM-1B-sft-bf16 --save-directory <converted_model_path>

:: to convert MiniCPM-2B-sft-bf16
python convert.py --repo-id-or-model-path openbmb/MiniCPM-2B-sft-bf16 --save-directory <converted_model_path>

:: to convert DeepSeek-R1-Distill-Qwen-1.5B
python convert.py --repo-id-or-model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  --save-directory <converted_model_path>

:: to convert DeepSeek-R1-Distill-Qwen-7B
python convert.py --repo-id-or-model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B  --save-directory <converted_model_path>
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the model (e.g.`Meta-llama/Llama-2-7b-chat-hf` for Llama2-7B) to be downloaded, or the path to the huggingface checkpoint folder.
- `--save-directory SAVE_DIRECTORY`: argument defining the path to save converted model. If it is a non-existing path, the original pretrained model specified by `REPO_ID_OR_MODEL_PATH` will be loaded, and the converted model will be saved into `SAVE_DIRECTORY`.
- `--max-context-len MAX_CONTEXT_LEN`: argument defining the maximum sequence length for both input and output tokens. It is default to be `1024`.
- `--max-prompt-len MAX_PROMPT_LEN`: argument defining the maximum number of tokens that the input prompt can contain. It is default to be `512`.
- `--low-bit LOW_BIT`: argument defining the low bit optimizations that will be applied to the model. Current available options are `"sym_int4"`, `"asym_int4"` and `"sym_int8"`, with `"sym_int4"` as the default.

## 3. Build C++ Example `llm-cli`(Optional)

- You can run below cmake script in cmd to build `llm-cli` by yourself, don't forget to replace below <CONDA_ENV_DIR> with your own path.

```cmd
:: under current directory
:: please replace below conda env dir with your own path
set CONDA_ENV_DIR=C:\Users\arda\miniforge3\envs\llm\Lib\site-packages
mkdir build
cd build
cmake ..
cmake --build . --config Release -j
cd Release
```

- You can also directly use our released `llm-cli.exe` which has the same usage as this example `llm-cli.cpp`

> [!NOTE]
> Our released `llm-cli.exe` can be found at <CONDA_ENV_DIR>\bigdl-core-npu

## 4. Run `llm-cli`

With built `llm-cli`, you can run the example with specified paramaters. For example,

```cmd
# Run simple text completion
llm-cli.exe -m <converted_model_path> -n 64 "AI是什么?"

# Run in conversation mode
llm-cli.exe -m <converted_model_path> -cnv
```

Arguments info:
- `-m` : argument defining the path of saved converted model.
- `-cnv` : argument to enable conversation mode.
- `-n` : argument defining how many tokens will be generated.
- Last argument is your input prompt.

### 5. Sample Output
#### [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
##### Text Completion
```cmd
AI stands for Artificial Intelligence, which is the field of study focused on creating and developing intelligent machines that can perform tasks that typically require human intelligence, such as visual and auditory recognition, speech recognition, and decision-making. AI is a broad and diverse field that includes a wide range

llm_perf_print:        load time =    xxxx.xx ms
llm_perf_print: prompt eval time =    xxxx.xx ms /    26 tokens (   xx.xx ms per token,    xx.xx tokens per second)
llm_perf_print:        eval time =    xxxx.xx ms /    63 runs   (   xx.xx ms per token,    xx.xx tokens per second)
llm_perf_print:       total time =    xxxx.xx ms /    89 tokens
```

##### Conversation
```cmd
User:Hi
Assistant: Hello! It's nice to meet you. How can I help you today?
User:What is AI in one sentence?
Assistant:Sure, here's a one-sentence definition of AI:

Artificial Intelligence (AI) refers to the development and use of computer systems and algorithms that can perform tasks that typically require human intelligence, such as visual and speech recognition, decision-making and problem-solving, and natural language processing.
User:exit
```

### Troubleshooting

#### Program crash with Chinese prompt
If you run CPP examples on Windows and find that your program raises below error when accepting Chinese prompts, you can search for `region` in the Windows search bar and go to `Region->Administrative->Change System locale..`, tick `Beta: Use Unicode UTF-8 for worldwide language support` option and then restart your computer.
```log
thread '<unnamed>' panicked at src\lib.rs:151:91:
called `Result::unwrap()` on an `Err` value: Utf8Error { valid_up_to: 77, error_len: Some(1) }
```

For detailed instructions on how to do this, see [this issue](https://github.com/intel-analytics/ipex-llm/issues/10989#issuecomment-2105598660).

#### Accuracy Tuning

If you enconter output issues when running the CPP examples, you could try the following methods [**when converting the model**](#2-convert-model) to tune the accuracy:

1. Before converting the model, consider setting an additional environment variable `IPEX_LLM_NPU_QUANTIZATION_OPT=1` to enhance output quality.

2. If you are using the default `LOW_BIT` value (i.e. `sym_int4` optimizations), you could try to use `--low-bit "asym_int4"` instead to tune the output quality.

3. You could refer to the [Quickstart](../../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#accuracy-tuning) for more accuracy tuning strategies.

> [!IMPORTANT]
> Please note that to make the above methods taking effect, you must specify a new folder for `SAVE_DIRECTORY`. Reusing the same `SAVE_DIRECTORY` will load the previously saved low-bit model, and thus making the above accuracy tuning strategies ineffective.