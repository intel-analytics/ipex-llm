# Migration Guide from BigDL-LLM

This guide helps you migrate your `bigdl-llm` application to use `ipex-llm`.


## Install/Upgrade
```eval_rst
.. note::

   This step assumes you have already installed `bigdl-llm`.
```
You need to uninstall `bigdl-llm` and install `ipex-llm`With your `bigdl-llm` conda envionment activated, exeucte the folloiwng command according to your device type and location (for GPU): 

### For CPU

```bash
pip uninstall -y bigdl-llm
pip install --pre --upgrade ipex-llm[all] 
```

### For GPU
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

## Using `ipex-llm`

### Compatibility Mode (Minimal Code Changes Required)

This approach allows you to retain your existing code with minimal changes. Simply add an extra line `import ipex_llm` at the start of your script.


```eval_rst
.. note::

   Remember to add `import ipex_llm` before any `bigdl.llm` imports.
```
```python
import ipex_llm # Add this line before any bigdl.llm imports
from bigdl.llm.transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             load_in_4bit=True,
                                             trust_remote_code=True)
```

### Full Migration

For a complete migration, change all `bigdl.llm` imports to `ipex_llm`. Replace lines such as `from bigdl.llm.transformers import AutoModelForCausalLM` with `from ipex_llm.transformers import AutoModelForCausalLM`, as illustrated below.

```python
# from bigdl.llm.transformers import AutoModelForCausalLM # Original line
from ipex_llm.transformers import AutoModelForCausalLM # Updated line
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             load_in_4bit=True,
                                             trust_remote_code=True)
```

