import transformers
import torch
import os
import intel_extension_for_pytorch as ipex
# verify hardware (how many gpu availables, gpu status, cpu info, memory info, etc.)
print("-"*20 + "GPU status" + "-"*20)
os.system("sudo xpu-smi stats -d 0")
os.system("lspci|grep 'VGA\|Display'")
# print("-"*20 + "CPU status" + "-"*20)
# os.system("htop")
print("-"*20 + "Memory Info" + "-"*20)
os.system("cat /proc/meminfo")
# verify software versions (transformer, bigdl, pytorch, ipex, one-api, driver)
print("-"*20 + "Transformer Version" + "-"*20)
print(transformers.__version__)
# pip show bigdl-llm
print("-"*20 + "BigDL Verson" + "-"*20)
os.system("pip show bigdl-llm")
print("-"*20 + "PyTorch Version" + "-"*20)
print(torch.__version__)
print("-"*20 + "IPEX Version" + "-"*20)
print(ipex.__version__)
# driver
print("-"*20 + "Driver Version" + "-"*20)
os.system("lspci")

