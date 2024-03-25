# bigdl-llm upgrade guide
## Upgrade bigdl-llm to ipex-llm
Need to uninstall bigdl-llm and install ipex-llm first.
```bash
pip uninstall -y bigdl-llm
pip install --pre --upgrade ipex-llm[all] # for cpu
pip install --pre --upgrade ipex-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu # for xpu
```
## Run bigdl-llm example code
Choose a way to modify the example code to use.

### 1.Run bigdl-llm example code with ipex-llm in compatible mode
Minimal change: Use `BIGDL_COMPATIBLE_MODE` to run, need to add `import ipex_llm` at the 
beginning of code with bigdl-llm examples.
```python
# need to add before import bigdl.llm
import ipex_llm
from bigdl.llm.transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             load_in_4bit=True,
                                             trust_remote_code=True)
```

### 2.Upgrade bigdl-llm example code to ipex-llm
Need to replace all `bigdl.llm` with `ipex_llm`.
```python
#from bigdl.llm.transformers import AutoModelForCausalLM
from ipex_llm.transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             load_in_4bit=True,
                                             trust_remote_code=True)
```

