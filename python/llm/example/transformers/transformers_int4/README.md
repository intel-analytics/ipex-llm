# BigDL-LLM Transformers INT4 Optimization for Large Language Model
You can use BigDL-LLM to run any Huggingface Transformer models with INT4 optimizations on either servers or laptops. This directory contains example scripts to help you quickly get started using BigDL-LLM to run some popular open-source models in the community. Each model has its own dedicated folder, where you can find detailed instructions on how to install and run it.

# Verified models
| Model     | Example                                                  |
|-----------|----------------------------------------------------------|
| LLaMA     | [link](vicuna)    |
| MPT       | [link](mpt)       |
| Falcon    | [link](falcon)    |
| ChatGLM   | [link](chatglm)   | 
| ChatGLM2  | [link](chatglm2)  | 
| MOSS      | [link](moss)      | 
| Baichuan  | [link](baichuan)  | 
| Dolly-v1  | [link](dolly_v1)  | 
| RedPajama | [link](redpajama) | 
| Phoenix   | [link](phoenix)   | 
| StarCoder | [link](starcoder) | 

## Recommended Requirements
To run the examples, we recommend using Intel® Xeon® processors (server), or >= 12th Gen Intel® Core™ processor (client).

For OS, BigDL-LLM supports Ubuntu 20.04 or later, CentOS 7 or later, and Windows 10/11.

## Best Known Configuration on Linux
For better performance, it is recommended to set environment variables on Linux with the help of BigDL-Nano:
```bash
pip install bigdl-nano
source bigdl-nano-init
```
