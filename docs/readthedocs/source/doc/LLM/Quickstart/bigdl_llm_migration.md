# `bigdl-llm` Migration Guide
## Upgrade `bigdl-llm` package to `ipex-llm`
First uninstall `bigdl-llm` and install `ipex-llm`.
```bash
pip uninstall -y bigdl-llm
pip install --pre --upgrade ipex-llm[all] # for cpu
pip install --pre --upgrade ipex-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu # for xpu
```
## Migrate `bigdl-llm` code to `ipex-llm`
There are two options to migrate `bigdl-llm` code to `ipex-llm`.

### 1. Upgrade `bigdl-llm` code to `ipex-llm`
To upgrade `bigdl-llm` code to `ipex-llm`, simply replace all `bigdl.llm` with `ipex_llm`:

```python
#from bigdl.llm.transformers import AutoModelForCausalLM
from ipex_llm.transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             load_in_4bit=True,
                                             trust_remote_code=True)
```

### 2. Run `bigdl-llm` code in compatible mode (experimental)
To run in the compatible mode, simply add `import ipex_llm` at the beginning of the existing `bigdl-llm` code:

```python
# need to add the below line before "import bigdl.llm"
import ipex_llm
from bigdl.llm.transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             load_in_4bit=True,
                                             trust_remote_code=True)
```

