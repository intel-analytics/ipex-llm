# Torch Graph Mode

Here, we provide how to run [torch graph mode](https://pytorch.org/blog/optimizing-production-pytorch-performance-with-graph-transformations/) on Intel Arc™ A-Series Graphics with ipex-llm, and [gpt2-medium](https://huggingface.co/openai-community/gpt2-medium) for classification task is used as illustration:

### 1. Install
```bash
conda create -n ipex-llm python=3.11
conda activate ipex-llm
pip install --pre --upgrade ipex-llm[xpu_arc] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/
pip install --pre pytorch-triton-xpu==3.0.0+1b2f15840e --index-url https://download.pytorch.org/whl/nightly/xpu
conda install -c conda-forge libstdcxx-ng
unset OCL_ICD_VENDORS
```

### 2. Configures OneAPI environment variables

> [!NOTE]
> Skip this step if you are running on Windows.

This is a required step on Linux for APT or offline installed oneAPI. Skip this step for PIP-installed oneAPI.

```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. Run

Convert text-generating GPT2-Medium to the classification:

   ```bash
   # The convert step needs to access the internet
   export http_proxy=http://your_proxy_url
   export https_proxy=http://your_proxy_url

   # This will yield gpt2-medium-classification under /llm/models in the container
   python convert-model-textgen-to-classfication.py --model-path MODEL_PATH
   ```

This will yield a mode directory ends with '-classification' neart your input model path.

Benchmark GPT2-Medium's performance with IPEX-LLM engine:

   ``` sbash
   ipexrun xpu gpt2-graph-mode-benchmark.py --device xpu --engine ipex-llm --batch 16 --model-path MODEL_PATH

   # You will see the key output like:
   # Average time taken (excluding the first two loops): xxxx seconds, Classification per seconds is xxxx
   ```
