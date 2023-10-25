# Auto Comment

In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on Qwen models to automatically generate comments for code files. For illustration purposes, we utilize the [Qwen/Qwen-VL-7B](https://huggingface.co/Qwen/Qwen-VL-7B) as a reference Qwen model.

> **Note**: If you want to download the Hugging Face *Transformers* model, please refer to [here](https://huggingface.co/docs/hub/models-downloading#using-git).
>
> BigDL-LLM optimizes the *Transformers* model in INT4 precision at runtime, and thus no explicit conversion is needed.

## Requirements
To run these examples with BigDL-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Generate comments for the provided code file
In the example [auto_comment.py](./auto_comment.py), we show a basic use case for a Qwen model to automatically generate comments for code file with `chat()` API, with BigDL-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://docs.conda.io/en/latest/miniconda.html#).

After installing conda, create a Python environment for BigDL-LLM:
```bash
conda create -n llm python=3.9 # recommend to use Python 3.9
conda activate llm

pip install --pre --upgrade bigdl-llm[all] # install the latest bigdl-llm nightly build with 'all' option
pip install tiktoken einops transformers_stream_generator  # additional package required for Qwen-7B-Chat to conduct generation
```

### 2. Run
After setting up the Python environment, you could run the example by following steps.

> **Note**: When loading the model in 4-bit, BigDL-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the ChatGLM model based on the capabilities of your machine.

#### 2.1 Client
On client Windows machines, it is recommended to run directly with full utilization of all cores:
```powershell
python ./auto_comment.py --path test_codes.py
```
More information about arguments can be found in [Arguments Info](#23-arguments-info) section. The expected output can be found in [Sample Output](#24-sample-output) section.

#### 2.2 Server
For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# set BigDL-Nano env variables
source bigdl-nano-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./auto_comment.py --path test_codes.py
```
More information about arguments can be found in [Arguments Info](#23-arguments-info) section. The expected output can be found in [Sample Output](#24-sample-output) section.

#### 2.3 Arguments Info
In the example, several arguments can be passed to satisfy your requirements:

- `--path`: str, the provided file to be commented. It can be a file (currently supports python code files) or a folder (all python code files in the folder will be scanned).
- `--repo-id-or-model-path`: str, argument defining the huggingface repo id for the Qwen model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'Qwen/Qwen-VL-7B'`.
- `--regenerate`: bool, argument defining whether comments need to be regenerated for the same file. It is default to be `False`.

#### 2.4 Sample Output
#### [Qwen/Qwen-VL-7B](https://huggingface.co/Qwen/Qwen-VL-7B) 

The content of sample input file `test_codes.py` is:
```Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)
data = np.random.randint(0, 100, (3, 10))
df = pd.DataFrame(data)
row_means = df.mean(axis=1)
filtered_df = df.where(df >= row_means[:, np.newaxis], 0)

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for i, row in enumerate(df.values):
    axs[i].hist(row, bins=5, alpha=0.7, label=f'{i + 1}')
plt.tight_layout()
plt.show()
```

The sample output file is named `test_codes_comment.py`, and the file content is as follows:
```Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置随机数种子，确保每次运行结果一致
np.random.seed(0)

# 生成一个3行10列的随机整数数组
data = np.random.randint(0, 100, (3, 10))

# 将数组转换为pandas DataFrame
df = pd.DataFrame(data)

# 计算每一行的平均值
row_means = df.mean(axis=1)

# 使用where函数筛选出大于平均值的元素为0，其余元素保留
filtered_df = df.where(df >= row_means[:, np.newaxis], 0)

# 创建一个包含3个子图的figure对象
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# 遍历DataFrame的每一行，使用hist函数生成直方图
for i, row in enumerate(df.values):
    axs[i].hist(row, bins=5, alpha=0.7, label=f'{i + 1}')

# 设置子图的间距
plt.tight_layout()

# 显示图形
plt.show()
```
