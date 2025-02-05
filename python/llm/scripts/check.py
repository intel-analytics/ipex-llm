#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import sys
import subprocess


def check_python_version():
    py_version = sys.version.split()[0]
    lst = py_version.split(".")
    v1 = eval(lst[0])
    v2 = eval(lst[1])
    if v1!=3 or v2<9:
        print("Python Version must be higher than 3.9.x, please check python version. More details could be found in the README.md")
        return 1
    return 0

def check_transformer_version():
    try:
        import transformers
        print(f"transformers={transformers.__version__}")
    except:
        print("Transformers is not installed.")

def check_torch_version():
    try:
        import torch
        print(f"torch={torch.__version__}")
    except:
        print("PyTorch is not installed.")

def check_ipex_llm_version():
    import os
    if os.system("pip show ipex-llm")!=0:
        print("ipex-llm is not installed")

def check_ipex_version():
    try:
        import intel_extension_for_pytorch as ipex
        print(f"ipex={ipex.__version__}")
    except:
        print("IPEX is not installed properly. ")


def check_memory():
    physical_mem = subprocess.run('wmic computersystem get totalphysicalmemory', capture_output=True, text=True).stdout
    """
    Example output:
        TotalPhysicalMemory
        68448202752
    """
    physical_mem = physical_mem.split('\n')
    for i in range(1, len(physical_mem)+1):
        if physical_mem[-i].strip().isdigit():
            print(f'Total Memory: {int(physical_mem[-i].strip()) / 1024**3:.3f} GB')
            break

    print()

    memory = subprocess.run('wmic memorychip get Capacity, Speed', capture_output=True, text=True).stdout
    """
    Example output:
        Capacity     Speed
        34359738368  3200
        34359738368  3200
    """
    memory = memory.split('\n\n')
    chip_count = 0
    for i in range(1, len(memory)+1):
        if memory[-i] != '' and 'Speed' not in memory[-i]:
            capacity, speed = memory[-i].strip().split('  ')

            # convert capacity from byte to GB
            capacity = str(int(int(capacity) / 1024**3))

            if capacity.isdigit() and speed.isdigit():
                print(f'Chip {chip_count} Memory: {capacity} GB | Speed: {speed} MHz')
                chip_count += 1

def check_cpu():
    cpu_info = subprocess.run('wmic cpu get Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed, Manufacturer', capture_output=True, text=True).stdout
    """
    Example output:
        Manufacturer  MaxClockSpeed  Name                                  NumberOfCores  NumberOfLogicalProcessors
        GenuineIntel  3200           12th Gen Intel(R) Core(TM) i9-12900K  16             24
    """

    cpu_info = cpu_info.split('\n\n')
    names = cpu_info[0]
    values = cpu_info[1]

    idx = 0
    while idx < len(names):
        if names[idx] != ' ':
            start = idx
            name_end = names[start:].find(' ') + start

            # for slicing the value
            value_end = name_end + 1
            while value_end < len(names) and names[value_end] == ' ':
                value_end += 1

            # updagte idx
            idx = value_end

            # get the slicing for the values
            value_end -= 2
            print(f'CPU {names[start:name_end]}: {values[start:value_end]}')            
        else:
            idx += 1

def check_gpu_driver():
    gpu_driver_info = subprocess.run("wmic path Win32_PnPSignedDriver where \"Description like '%Intel%' and Description like '%Graphics%'\" get DeviceName, DriverVersion", capture_output=True, text=True).stdout
    """
    Example output:
        DeviceName                                                        DriverVersion
        Intel(R) Graphics System Controller Auxiliary Firmware Interface  2322.4.7.0
        Intel(R) Graphics System Controller Firmware Interface            2337.5.3.0
        Intel(R) Graphics Command Center                                  31.0.101.5084
        Intel(R) Arc(TM) A770 Graphics                                    31.0.101.5084
    """
    gpu_driver_info = gpu_driver_info.split('\n\n')
    driver_info = []

    # filter the information
    for i in gpu_driver_info[1:]:
        if i != '':
            if 'Controller' not in i and 'Command Center' not in i:
                driver_info.append(i.strip())
    
    # print the gpu driver info
    gpu_num = 0
    for gpu in driver_info:
        splitted = gpu.split('  ')
        print(f'GPU {gpu_num}: {splitted[0]} \t Driver Version: {splitted[-1]}')
        gpu_num += 1

def main():
    if check_python_version()!=0:
        return -1
    print("-----------------------------------------------------------------")
    check_transformer_version()
    print("-----------------------------------------------------------------")
    check_torch_version()
    print("-----------------------------------------------------------------")
    check_ipex_llm_version()
    print("-----------------------------------------------------------------")
    check_ipex_version()
    print("-----------------------------------------------------------------")
    check_memory()
    print("-----------------------------------------------------------------")
    check_cpu()
    print("-----------------------------------------------------------------")
    check_gpu_driver()
    print("-----------------------------------------------------------------")

if __name__ == "__main__":
    main()
