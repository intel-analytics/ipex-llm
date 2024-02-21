# README
Script will first check Python installation, then site-packages (transformers, torch and bigdl-llm), and hardware or operating system and finally check xpu-related issue. 
## Linux
### How to use
Run `bash env-check.sh` on linux. 
### Sample output
Here is the output of the scipt on arc12 with GPU installed properly.
Note that when only CPU is aviailable, we can ignore xpu related checks. 
```
-----------------------------------------------------------------
PYTHON_VERSION=3.9.18
-----------------------------------------------------------------
transformers=4.37.0
-----------------------------------------------------------------
torch=2.1.0a0+cxx11.abi
-----------------------------------------------------------------
BigDL Version: 2.5.0b20240219
-----------------------------------------------------------------
CPU Information: 
Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bit
Address sizes:                   46 bits physical, 48 bits virtual
Byte Order:                      Little Endian
CPU(s):                          32
On-line CPU(s) list:             0-31
Vendor ID:                       GenuineIntel
Model name:                      13th Gen Intel(R) Core(TM) i9-13900K
-----------------------------------------------------------------
MemTotal:       65585208 kB
-----------------------------------------------------------------
ulimit: 
......
......
-----------------------------------------------------------------
Operating System: 
Ubuntu 22.04.3 LTS \n \l

-----------------------------------------------------------------
Environment Variable: 
......
......
......
-----------------------------------------------------------------
xpu-smi is properly installed. 
-----------------------------------------------------------------
+-----------+--------------------------------------------------------------------------------------+
| Device ID | Device Information                                                                   |
+-----------+--------------------------------------------------------------------------------------+
| 0         | Device Name: Intel(R) Arc(TM) A770 Graphics                                          |
|           | Vendor Name: Intel(R) Corporation                                                    |
|           | SOC UUID: 00000000-0000-0003-0000-000856a08086                                       |
|           | PCI BDF Address: 0000:03:00.0                                                        |
|           | DRM Device: /dev/dri/card0                                                           |
|           | Function Type: physical                                                              |
+-----------+--------------------------------------------------------------------------------------+
-----------------------------------------------------------------
ipex=2.1.10+xpu
-----------------------------------------------------------------
```

### FAQs
1. How to create environment with proper python version and dependencies?
We suggest using conda to manage environment:
When CPU-only, 
```bash
conda create -n llm python=3.9
conda activate llm
pip install bigdl-llm[all] # install bigdl-llm with 'all' option
```
When XPU is available
```bash
conda create -n llm python=3.9
conda activate llm
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
```
If you get messages "Python Version must be higher than 3.9.x" or "No Python found!", we recommend install the virtual environment again with the above commands.

2. How to install xpu-smi?
TODO

3. Site-package version. Here we list transformers, torch, BigDL. 