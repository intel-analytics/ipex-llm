# Fuyu
In this directory, you will find examples on how you could apply IPEX-LLM INT4 optimizations on Fuyu models. For illustration purposes, we utilize the [adept/fuyu-8b](https://huggingface.co/adept/fuyu-8b) as a reference Fuyu model.

## Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for an Fuyu model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://docs.conda.io/en/latest/miniconda.html#).

After installing conda, create a Python environment for IPEX-LLM:
```bash
conda create -n llm python=3.11 # recommend to use Python 3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all] # install the latest ipex-llm nightly build with 'all' option

pip install transformers==4.35 pillow # additional package required for Fuyu to conduct generation
```

### 2. Run
After setting up the Python environment, you could run the example by following steps.

> **Note**: When loading the model in 4-bit, IPEX-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the Fuyu model based on the capabilities of your machine.

#### 2.1 Client
On client Windows machines, it is recommended to run directly with full utilization of all cores:
```powershell
python ./generate.py --image-path demo.jpg
```
More information about arguments can be found in [Arguments Info](#23-arguments-info) section. The expected output can be found in [Sample Output](#24-sample-output) section.

#### 2.2 Server
For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# set IPEX-LLM env variables
source ipex-llm-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./generate.py --image-path demo.jpg
```
More information about arguments can be found in [Arguments Info](#23-arguments-info) section. The expected output can be found in [Sample Output](#24-sample-output) section.

#### 2.3 Arguments Info
In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Fuyu model (e.g. `adept/fuyu-8b`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'adept/fuyu-8b'`.
- `--prompt PROMPT`: argument defining the prompt to be inferred (with the image for chat). It is default to be `'Generate a coco-style caption.'`.
- `--image-path IMAGE_PATH`: argument defining the input image that the chat will focus on. It is required and should be a local path (not url).
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `512`.


#### 2.4 Sample Output
#### [adept/fuyu-8b](https://huggingface.co/adept/fuyu-8b)

```log
Inference time: xxxx s
-------------------- Prompt --------------------
Generate a coco-style caption.
-------------------- Output --------------------
An orange bus parked on the side of a road.
```

The sample input image is (which is fetched from [COCO dataset](https://cocodataset.org/#explore?id=178242)):

[demo.jpg](https://cocodataset.org/#explore?id=178242)

<a href="http://farm6.staticflickr.com/5331/8954873157_539393fece_z.jpg"><img width=400px src="http://farm6.staticflickr.com/5331/8954873157_539393fece_z.jpg" ></a>