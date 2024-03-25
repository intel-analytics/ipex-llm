# IPEX-LLM INT4 Optimization for Large Language Model
You can use `optimize_model` API to accelerate general PyTorch models on Intel servers and PCs. This directory contains example scripts to help you quickly get started using IPEX-LLM to run some popular open-source models in the community. Each model has its own dedicated folder, where you can find detailed instructions on how to install and run it.

## Recommended Requirements
To run the examples, we recommend using Intel® Xeon® processors (server), or >= 12th Gen Intel® Core™ processor (client).

For OS, IPEX-LLM supports Ubuntu 20.04 or later, CentOS 7 or later, and Windows 10/11.

## Best Known Configuration on Linux
For better performance, it is recommended to set environment variables on Linux with the help of IPEX-LLM:
```bash
pip install ipex-llm
source ipex-llm-init
```
