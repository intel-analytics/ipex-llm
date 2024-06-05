# Bark
In this directory, you will find examples on how you could use IPEX-LLM `optimize_model` API to accelerate Bark models. For illustration purposes, we utilize the [suno/bark](https://huggingface.co/suno/bark) as reference Bark models.

## Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Synthesize speech with the given input text
In the example [synthesize_speech.py](./synthesize_speech.py), we show a basic use case for Bark model to synthesize speech based on the given text, with IPEX-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://conda-forge.org/download/).

After installing conda, create a Python environment for IPEX-LLM:


On Linux:

```bash
conda create -n llm python=3.11 # recommend to use Python 3.11
conda activate llm

# install the latest ipex-llm nightly build with 'all' option
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
pip install TTS scipy
```

On Windows:

```cmd
conda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all]
pip install TTS scipy
```

### 2. Download Bark model
Before running the example, you need to download Bark model to local folder:
```python
from huggingface_hub import snapshot_download

model_path = snapshot_download(repo_id='suno/bark',
                               local_dir='bark/') # you can change `local_dir` parameter to specify any local folder
```

Please refer to [here](https://huggingface.co/docs/huggingface_hub/guides/download#download-files-to-local-folder) for more information about `snapshot_download`.

### 3. Run
After setting up the Python environment and downloading Bark model, you could run the example by following steps.

#### 3.1 Client
On client Windows machines, it is recommended to run directly with full utilization of all cores:
```cmd
# make sure `--model-path` corresponds to the local folder of downloaded model
python ./synthesize_speech.py --model-path 'bark/' --text "This is an example text for synthesize speech."
```
More information about arguments can be found in [Arguments Info](#33-arguments-info) section.

#### 3.2 Server
For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# set IPEX-LLM env variables
source ipex-llm-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
# make sure `--model-path` corresponds to the local folder of downloaded model
numactl -C 0-47 -m 0 python ./synthesize_speech.py --model-path 'bark/' --text "This is an example text for synthesize speech."
```
More information about arguments can be found in [Arguments Info](#33-arguments-info) section.

#### 3.3 Arguments Info
In the example, several arguments can be passed to satisfy your requirements:

- `--model-path MODEL_PATH`: **required**, argument defining the local path to the Bark model checkpoint folder.
- `--text TEXT`: argument defining the text to synthesize speech. It is default to be `"This is an example text for synthesize speech."`.
