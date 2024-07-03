# `bigdl-llm` Migration Guide

This guide helps you migrate your `bigdl-llm` application to use `ipex-llm`.

## Table of Contents
- [Upgrade `bigdl-llm` package to `ipex-llm`](./bigdl_llm_migration.md#1-upgrade-bigdl-llm-code-to-ipex-llm)
- [Migrate `bigdl-llm` code to `ipex-llm`](./bigdl_llm_migration.md#migrate-bigdl-llm-code-to-ipex-llm)


## Upgrade `bigdl-llm` package to `ipex-llm`

> [!NOTE]
> This step assumes you have already installed `bigdl-llm`.

You need to uninstall `bigdl-llm` and install `ipex-llm`With your `bigdl-llm` conda environment activated, execute the following command according to your device type and location:

### For CPU

```bash
pip uninstall -y bigdl-llm
pip install --pre --upgrade ipex-llm[all] # for cpu
```

### For GPU
Choose either US or CN website for `extra-index-url`:

- For **US**:

  ```bash
  pip uninstall -y bigdl-llm
  pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
  ```

- For **CN**:

  ```bash
  pip uninstall -y bigdl-llm
  pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/
  ```

## Migrate `bigdl-llm` code to `ipex-llm`
There are two options to migrate `bigdl-llm` code to `ipex-llm`.

### 1. Upgrade `bigdl-llm` code to `ipex-llm`
To upgrade `bigdl-llm` code to `ipex-llm`, simply replace all `bigdl.llm` with `ipex_llm`:

```python
#from bigdl.llm.transformers import AutoModelForCausalLM # Original line
from ipex_llm.transformers import AutoModelForCausalLM #Updated line
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             load_in_4bit=True,
                                             trust_remote_code=True)
```

### 2. Run `bigdl-llm` code in compatible mode (experimental)
To run in the compatible mode, simply add `import ipex_llm` at the beginning of the existing `bigdl-llm` code:

```python
import ipex_llm # Add this line before any bigdl.llm imports
from bigdl.llm.transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             load_in_4bit=True,
                                             trust_remote_code=True)
```
