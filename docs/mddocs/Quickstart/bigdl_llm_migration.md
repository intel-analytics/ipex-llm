# `bigdl-llm` Migration Guide

This guide helps you migrate your `bigdl-llm` application to use `ipex-llm`.

## Upgrade `bigdl-llm` package to `ipex-llm`

```eval_rst
.. note::
   This step assumes you have already installed `bigdl-llm`.
```
You need to uninstall `bigdl-llm` and install `ipex-llm`With your `bigdl-llm` conda environment activated, execute the following command according to your device type and location:

### For CPU

```bash
pip uninstall -y bigdl-llm
pip install --pre --upgrade ipex-llm[all] # for cpu
```

### For GPU
Choose either US or CN website for `extra-index-url`:
```eval_rst
.. tabs::

   .. tab:: US

      .. code-block:: cmd

         pip uninstall -y bigdl-llm
         pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

   .. tab:: CN

      .. code-block:: cmd

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
