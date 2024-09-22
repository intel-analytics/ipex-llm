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
import platform

#import torch 
#import intel_extension_for_pytorch
#device = torch.xpu.get_device_name()

mode = sys.argv[1]
device = ""
tmpfilepath = sys.argv[2]

def get_xpu_device_type(device_name):
    #name = torch.xpu.get_device_name(x.device.index)
    if device_name.startswith("Intel(R) Arc(TM) A"):
        return "Arc"
    elif device_name.startswith("Intel(R) Arc(TM)"):
        return "iGPU"
    elif device_name.startswith("Intel(R) Data Center GPU Flex"):
        return "others"
    elif device_name.startswith("Intel(R) Data Center GPU Max"):
        return "others"
    elif device_name.startswith("Intel(R) UHD"):
        return "iGPU"
    elif device_name.startswith("Intel(R) Iris(R) Xe Graphics"):
        return "iGPU"
    else:
        return "others"



def let_user_select(option_list: list[str], queryname: str, default : int = 0, info: list[str] = None):
    if len(option_list) == 0:
        return default
    while True:
        if info is not None:
            for line in info:
                print(line)
        print(f"Please select a {queryname} (default: {default} {option_list[default]}):")
        for i, option in enumerate(option_list):
            print(f"{i}: {option}")
        userinput = input("> ")
        if userinput == "":
            return 0
        if userinput.isdigit():
            selected_id = int(userinput)
            if selected_id >= 0 and selected_id < len(option_list):
                return selected_id
            print("Invalid input: your input is out of range.")
        print("Invalid input: your input is not a number.")
            

def get_device_name():
    try:
        import torch
        import intel_extension_for_pytorch
    except ImportError:
        print("ERROR: Cannot import torch or intel_extension_for_pytorch")
        sys.exit(1)

    devicecount = torch.xpu.device_count()
    device_list = [torch.xpu.get_device_name(i) for i in range(devicecount)]
    if len(device_list) == 0:
        print("ERROR: No XPU devices found")
        sys.exit(1)
    
    if len(device_list) == 1:
        device = device_list[0]
        device_id = 0
        print(f"Selected device: {device}")
        print(f"Device ID: {device_id}")
        device = get_xpu_device_type(device)
        print(f"Device type: {device}")    
        return device, device_id
    
    device_select_info = []
    if platform.system() == "Windows":
        device_select_info.append("Multiple XPU devices found")
        device_select_info.append("Only one XPU device mode is supported on windows at the moment, please select the device you want to use.")
        device_select_info.append(" ")
        device_select_info.append("Warning : Have multiple XPU devices on windows may cause 'RuntimeError: could not create a primitive'")
        device_select_info.append("""Check https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Overview/FAQ/faq.html#runtimeerror-could-not-create-a-primitive-on-windows to find how to solve this problem.""")

    device_id = let_user_select(device_list, "device", 0, device_select_info)
    device = device_list[device_id]
    print(f"Selected device: {device}")
    print(f"Device ID: {device_id}")
    device = get_xpu_device_type(device)
    print(f"Device type: {device}")    
    return device, device_id



if mode == "select":
    modelist = ["cpu", "gpu"]
    infolist = []
    infolist.append("Please select the mode you want to use")
    infolist.append("The cpu mode option will enable cpu support, while the gpu mode option enable gpu support.")
    infolist.append("If you choose gpu mode, please ensure you have torch and intel-extension-for-pytorch in your python environment.")
    mode = modelist[let_user_select(modelist, "mode", 0, infolist)]



if mode == "auto":
    try:
        import torch
        import intel_extension_for_pytorch
    except ImportError:
        print("ERROR: Cannot import torch or intel_extension_for_pytorch")
        sys.exit(1)

    device = torch.xpu.get_device_name()
    if device == "":
        mode = "cpu"
    else:
        mode = "gpu"
    
if mode == "cpu":
    pass
elif mode == "gpu":
    device, device_id = get_device_name()



ONEAPI_DEVICE_SELECTOR = ""
SYCL_CACHE_PERSISTENT = ""
BIGDL_LLM_XMX_DISABLED = ""

if mode == "gpu":
    ONEAPI_DEVICE_SELECTOR = f"level_zero:{device_id}"
    if device == "iGPU":
        SYCL_CACHE_PERSISTENT = "1"
        BIGDL_LLM_XMX_DISABLED = "1"
    if device == "Arc":
        SYCL_CACHE_PERSISTENT = "1"
    

with open(tmpfilepath, "w") as f:
    f.write(f"{ONEAPI_DEVICE_SELECTOR}\n")
    f.write(f"{SYCL_CACHE_PERSISTENT}\n")
    f.write(f"{BIGDL_LLM_XMX_DISABLED}\n")
