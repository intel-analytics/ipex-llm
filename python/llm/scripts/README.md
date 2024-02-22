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
ipex=2.1.10+xpu
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
real-time non-blocking time  (microseconds, -R) unlimited
core file size              (blocks, -c) 0
data seg size               (kbytes, -d) unlimited
scheduling priority                 (-e) 0
file size                   (blocks, -f) unlimited
pending signals                     (-i) 255907
max locked memory           (kbytes, -l) 8198148
max memory size             (kbytes, -m) unlimited
open files                          (-n) 1048576
pipe size                (512 bytes, -p) 8
POSIX message queues         (bytes, -q) 819200
real-time priority                  (-r) 0
stack size                  (kbytes, -s) 8192
cpu time                   (seconds, -t) unlimited
max user processes                  (-u) 255907
virtual memory              (kbytes, -v) unlimited
file locks                          (-x) unlimited
-----------------------------------------------------------------
Operating System: 
Ubuntu 22.04.3 LTS \n \l

-----------------------------------------------------------------
Environment Variable: 
SHELL=/usr/bin/zsh
LSCOLORS=Gxfxcxdxbxegedabagacad
TBBROOT=/opt/intel/oneapi/tbb/2021.11/env/..
USER_ZDOTDIR=/home/arda
COLORTERM=truecolor
LESS=-R
TERM_PROGRAM_VERSION=1.86.2
ONEAPI_ROOT=/opt/intel/oneapi
CONDA_EXE=/home/arda/anaconda3/bin/conda
_CE_M=
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
```

### FAQs
1. How to create environment with proper python version and dependencies?
We suggest using conda to manage environment:
When CPU-only. Note that after creating environment by commands below, running `bash env-check.sh` will get `ipex not installed` and that is normal. 
```bash
conda create -n llm python=3.9
conda activate llm
pip install bigdl-llm[all] # install bigdl-llm with 'all' option
```
When XPU is available. Note that after creating environment by commands below, running `bash env-check.sh` will get ipex version if installed correctly. 
```bash
conda create -n llm python=3.9
conda activate llm
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
```
If you get messages "Python Version must be higher than 3.9.x" or "No Python found!", we recommend install the virtual environment again with the above commands.

2. How to install xpu-smi?
TODO

3. Site-package version. Here we list transformers, torch, BigDL and ipex. 

## Windows
### How to use
Run `env-check.bat` on linux. 
### Sample output
Here is the output of the scipt on the laptop with GPU installed properly.
Note that when only CPU is aviailable, we can ignore xpu related checks. 
```
Python 3.9.18
-----------------------------------------------------------------
transformers=4.37.0
-----------------------------------------------------------------
torch=2.1.0a0+cxx11.abi
-----------------------------------------------------------------
Name: bigdl-llm
Version: 2.5.0b20240220
Summary: Large Language Model Develop Toolkit
Home-page: https://github.com/intel-analytics/BigDL
Author: BigDL Authors
Author-email: bigdl-user-group@googlegroups.com
License: Apache License, Version 2.0
Location: c:\users\zhicunlv\appdata\local\miniconda3\envs\zhicun\lib\site-packages
Requires:
Required-by:
-----------------------------------------------------------------
ipex=2.1.10+xpu
-----------------------------------------------------------------
System Information

Host Name:                 ZHICUNLV-MOBL
OS Name:                   Microsoft Windows 11 Enterprise
OS Version:                10.0.22621 N/A Build 22621
OS Manufacturer:           Microsoft Corporation
OS Configuration:          Member Workstation
OS Build Type:             Multiprocessor Free
Registered Owner:          Intel User
Registered Organization:   Intel Corporation
Product ID:                00330-80000-00000-AA731
Original Install Date:     1/17/2024, 1:48:03 AM
System Boot Time:          2/21/2024, 12:15:27 PM
System Manufacturer:       HP
System Model:              HP EliteBook 840 G8 Notebook PC
System Type:               x64-based PC
Processor(s):              1 Processor(s) Installed.
                           [01]: Intel64 Family 6 Model 140 Stepping 1 GenuineIntel ~2995 Mhz
BIOS Version:              HP T76 Ver. 01.15.02, 11/15/2023
Windows Directory:         C:\windows
System Directory:          C:\windows\system32
Boot Device:               \Device\HarddiskVolume1
System Locale:             en-us;English (United States)
Input Locale:              zh-cn;Chinese (China)
Time Zone:                 (UTC+08:00) Beijing, Chongqing, Hong Kong, Urumqi
Total Physical Memory:     32,496 MB
Available Physical Memory: 16,528 MB
Virtual Memory: Max Size:  37,360 MB
Virtual Memory: Available: 14,926 MB
Virtual Memory: In Use:    22,434 MB
Page File Location(s):     C:\pagefile.sys
Domain:                    ccr.corp.intel.com
Logon Server:              \\SHSCCR603
Hotfix(s):                 3 Hotfix(s) Installed.
                           [01]: KB5034467
                           [02]: KB5034765
                           [03]: KB5034225
Network Card(s):           3 NIC(s) Installed.
                           [01]: Intel(R) Wi-Fi 6 AX201 160MHz
                                 Connection Name: Wi-Fi
                                 DHCP Enabled:    Yes
                                 DHCP Server:     1.1.1.1
                                 IP address(es)
                                 [01]: 10.238.0.86
                                 [02]: fe80::4d61:f10d:d482:8bad
                           [02]: Bluetooth Device (Personal Area Network)
                                 Connection Name: Bluetooth Network Connection
                                 Status:          Media disconnected
                           [03]: Realtek USB GbE Family Controller
                                 Connection Name: Ethernet 2
                                 Status:          Media disconnected
Hyper-V Requirements:      A hypervisor has been detected. Features required for Hyper-V will not be displayed.
-----------------------------------------------------------------
'xpu-smi.exe' is not recognized as an internal or external command,
operable program or batch file.
xpu-smi is not installed properly.
```