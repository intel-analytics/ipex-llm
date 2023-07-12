# BigDL-LLM Transformers INT4 Optimization for Large Language Model
BigDL-LLM supports INT4 optimizations to any Hugging Face Transformers models.

In this directory, we provide several model-specific examples with BigDL-LLM INT4 optimizations. You can navigate to the folder corresponding to the model you want to use.

We are adding more and more model-specific examples.
## Recommended Requirements
To run the examples, we recommend using Intel® Xeon® processors (server), or >= 12th Gen Intel® Core™ processor (client).

For OS, BigDL-LLM supports Ubuntu 20.04 or later, CentOS 7 or later, and Windows 10/11.

## Best Known Configuration
For better performance, it is recommended to set environment variables with the help of BigDL-Nano:
```bash
pip install bigdl-nano
```
following with
| Linux | Windows (powershell)|
|:------|:-------|
|`source bigdl-nano-init`|`bigdl-nano-init`|
