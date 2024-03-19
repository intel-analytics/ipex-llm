# Langchain examples

The example here show how to use langchain RAG with `bigdl-llm` on intel GPU. This script rag2.py can analyze PDF file.
The example is adapted from [langchain cookbook](https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_structured_multi_modal_RAG_LLaMA2.ipynb).

## Install bigdl-llm
Follow the instructions in [Install](https://github.com/intel-analytics/BigDL/tree/main/python/llm#install).

## Install Required Dependencies for langchain examples. 

```bash
pip install langchain unstructured[all-docs] lxml
pip install pydantic==1.10.9
pip install -U chromadb==0.3.25
pip install -U pandas==2.0.3
sudo apt-get update
sudo apt-get install tesseract-ocr
pip install gpt4all
```

## Configures OneAPI environment variables
### Configurations for Linux
```bash
source /opt/intel/oneapi/setvars.sh
```
### Configurations for Windows
```cmd
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```
> Note: Please make sure you are using **CMD** (**Anaconda Prompt** if using conda) to run the command as PowerShell is not supported.

## Runtime Configurations
For optimal performance, it is recommended to set several environment variables. Please check out the suggestions based on your device.
### Configurations for Linux
<details>

<summary>For Intel Arc™ A-Series Graphics and Intel Data Center GPU Flex Series</summary>

```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```

</details>

<details>

<summary>For Intel Data Center GPU Max Series</summary>

```bash
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export ENABLE_SDP_FUSION=1
```
> Note: Please note that `libtcmalloc.so` can be installed by `conda install -c conda-forge -y gperftools=2.10`.
</details>

### Configurations for Windows
<details>

<summary>For Intel iGPU</summary>

```cmd
set SYCL_CACHE_PERSISTENT=1
set BIGDL_LLM_XMX_DISABLED=1
```

</details>

<details>

<summary>For Intel Arc™ A300-Series or Pro A60</summary>

```cmd
set SYCL_CACHE_PERSISTENT=1
```

</details>

<details>

<summary>For other Intel dGPU Series</summary>

There is no need to set further environment variables.

</details>

> Note: For the first time that each model runs on Intel iGPU/Intel Arc™ A300-Series or Pro A60, it may take several minutes to compile.

## Run the examples

### Semi Structured Multi Modal RAG

Before running the example, please download `punkt` in Python console panel:
```python
import nltk
nltk.download('punkt')
```

```bash
python rag2.py -m MODEL_PATH -p PDF_PATH -q QUESTION
```
arguments info:
- `-m MODEL_PATH`: **required**, path to the model
- `-p PDF_PATH`: **required**, path to the PDF
- `-q QUESTION`: question to ask. Default is `What is the method of this paper?`.

#### Known Issues
##### 1
When you have SSL issues related to huggingface.co, you can try using mirror site:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
##### 2
When you run the script on Windows, you may need to set up tesseracr manually by following steps:
1. Download tesseract exe from https://github.com/UB-Mannheim/tesseract/wiki.
2. Install this exe in `C:\Program Files (x86)\Tesseract-OCR`
3. Open virtual machine command prompt in windows or anaconda prompt.
4. Run `pip install pytesseract` and `set PATH=%PATH%;C:\Program Files (x86)\Tesseract-OCR`
5. To test if tesseract is installed type in python prompt:
   import pytesseract
   print(pytesseract)

## Example Output
```
Answer: The method of this paper is the development of a conditional diffusion model for generating Chinese calligraphy. The model uses a U-net model as the backbone and employs DDPMs sampling for the forward process (diffusion) and the reverse process (denoising). The style transfer technique is effective, and the system can generate artworks of any character, any script, and any style. The model can also perform style transfer via one-shot fine-tuning, which allows the transfer of scripts and styles to unseen characters and out-of-domain symbols. The fine-tuning technique is based on LoRA, which speeds up the training process of large models while reducing memitations of traditional learning methods and providing 
```

