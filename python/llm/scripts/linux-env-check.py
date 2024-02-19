import transformers
import torch
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="env-info")
    parser.add_argument("--xpu", action='store_true', help="specify whether use intel XPU environment")
    args = parser.parse_args()
    
    # verify hardware (how many gpu availables, gpu status, cpu info, memory info, etc.)
    print("-"*20 + "CPU status" + "-"*20)
    os.system("sycl-ls | grep cpu")
    print("-"*20 + "Memory Info" + "-"*20)
    os.system("cat /proc/meminfo | grep MemTotal")
    if args.xpu:
        import intel_extension_for_pytorch as ipex
        print("-"*20 + "GPU status" + "-"*20)
        os.system("sudo xpu-smi discovery")
    
        

    # verify software versions (transformer, bigdl, pytorch, ipex, one-api, driver)
    print("-"*20 + "Python" + "-"*20)
    os.system("python --version")
    print("-"*20 + "Transformer Version" + "-"*20)
    print(transformers.__version__)
    print("-"*20 + "BigDL Verson" + "-"*20)
    os.system("pip show bigdl-llm | grep Version:")
    print("-"*20 + "PyTorch Version" + "-"*20)
    print(torch.__version__)
    if args.xpu:
        print("-"*20 + "IPEX Version" + "-"*20)
        print(ipex.__version__)
    print("-"*20 + "Driver Version" + "-"*20)
    os.system("lspci")
    
    # verify os, one-api, and other env configurations (env variables, ulimit, numa, threads, etc.)
    print("-"*20 + "Operating System" + "-"*20)
    os.system("cat /proc/version")
    os.system("cat /etc/issue")
    print("-"*20 + "Env" + "-"*20)
    os.system("printenv")
    print("-"*20 + "Ulimit" + "-"*20)
    os.system("ulimit -a")
