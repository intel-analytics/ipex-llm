# IPEX-LLM Transformers INT4 Optimization for Large Language Model
You can use IPEX-LLM to run any Huggingface Transformer models with INT4 optimizations on either servers or laptops. This directory contains example scripts to help you quickly get started using IPEX-LLM to run some popular open-source models in the community. Each model has its own dedicated folder, where you can find detailed instructions on how to install and run it.

## Recommended Requirements
To run the examples, we recommend using Intel® Xeon® processors (server), or >= 12th Gen Intel® Core™ processor (client).

For OS, IPEX-LLM supports Ubuntu 20.04 or later (glibc>=2.17), CentOS 7 or later (glibc>=2.17), and Windows 10/11.

## Best Known Configuration on Linux
For better performance, it is recommended to set environment variables on Linux with the help of IPEX-LLM:
```bash
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
source ipex-llm-init
```
