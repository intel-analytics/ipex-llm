#  Utility Scripts


## Env-Check

The **Env-Check** scripts  ([env-check.sh](./env-check.sh), [env-check.bat](./env-check.bat)) are designed to verify your `ipex-llm` installation and runtime environment. These scripts can help you ensure your environment is correctly set up for optimal performance. You can include the script's output when reporting issues on [IPEX Github Issues](https://github.com/intel-analytics/ipex-llm/issues) for easier troubleshooting.

> Note: These scripts verify python installation, check for necessary packages and environmental variables, assess hardware or operating system compatibility, and identify any XPU-related issues. 

### Install extra dependency

* On Linux, the script uses a tool named `xpu-smi`. It is a convinent tool the monitor ths status of the GPUs. If you're runing LLMs on GPUs on Linux, we recommend installing `xpu-smi`. Run below command to install:
```
sudo apt install xpu-smi
```
* On Windows, you can ignore the sections in `xpu-smi.exe` if you didn't install it. You can always use **Windows Task Manager** to monitor the status of GPUs on Windows.
  
### Usage

* After installing `ipex-llm`, open a terminal (on Linux) or **Miniforge Prompt** (on Windows), and activate the conda environment you have created for running `ipex-llm`: 
  ```
  conda activate llm
  ```
  > If you do not know how to install `ipex-llm`, refer to [IPEX-LLM installation](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Overview/install.html) for more details.
*  Within the activated python environment, run below command:
    *  On Linux
        1. Install clinfo 
          ```sudo apt install clinfo```

        2. Activate oneapi
            Activate the `setvars.sh` file in the folder where you installed the oneapi
            
            ```
            source /opt/intel/oneapi/setvars.sh
            ```

        3. Run the env check
            ```bash
            bash env-check.sh
            ```

    * On Windows
        1. Activate oneapi
            Activate the `setvars.bat` file in the folder where you installed the oneapi

            ```bash
            call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
            ```

        2. Download the XPU manager
            Go to the [xpu manager download webpage](https://github.com/intel/xpumanager/releases) to download the latest `xpu-smi` zip file (e.g. xpu-smi-1.2.34-20240417.060819.a50c0371_win.zip). Unzip it and copy the `env-check.bat` and `check.py` files into the unzipped folder.

        3. Run the env check
            In your terminal, enter the unzipped folder and run:
            ```bash
            env-check.bat
            ```
        
        4. Additional Information
            If you want to know the GPU memory information, you can use `ctrl+shift+esc` to open the task manager.
            Then enter the performance section on the left navigation bar and go to the GPU section,
            you can check GPU memory under the `GPU Memory`.

### Sample outputs

* Linux Desktop equipped with i9-13900K Intel Core CPU and Intel(R) Arc(TM) A770 GPU example output:

```
-----------------------------------------------------------------
PYTHON_VERSION=3.11.9
-----------------------------------------------------------------
transformers=4.31.0
-----------------------------------------------------------------
torch=2.1.0a0+cxx11.abi
-----------------------------------------------------------------
ipex-llm Version: 2.1.0b20240506
-----------------------------------------------------------------
ipex=2.1.10+xpu
-----------------------------------------------------------------
CPU Information: 
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Address sizes:                      46 bits physical, 48 bits virtual
Byte Order:                         Little Endian
CPU(s):                             32
On-line CPU(s) list:                0-31
Vendor ID:                          GenuineIntel
Model name:                         13th Gen Intel(R) Core(TM) i9-13900K
CPU family:                         6
Model:                              183
Thread(s) per core:                 2
Core(s) per socket:                 24
Socket(s):                          1
Stepping:                           1
CPU max MHz:                        5800.0000
CPU min MHz:                        800.0000
BogoMIPS:                           5990.40
-----------------------------------------------------------------
Total CPU Memory: 62.5306 GB
-----------------------------------------------------------------
Operating System: 
Ubuntu 22.04.4 LTS \n \l

-----------------------------------------------------------------
Linux arda-arc09 6.5.0-28-generic #29~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Thu Apr  4 14:39:20 UTC 2 x86_64 x86_64 x86_64 GNU/Linux
-----------------------------------------------------------------
CLI:
    Version: 1.2.31.20240308
    Build ID: 00000000

Service:
    Version: 1.2.31.20240308
    Build ID: 00000000
    Level Zero Version: 1.16.0
-----------------------------------------------------------------
  Driver Version                                  2023.16.12.0.12_195853.xmain-hotfix
  Driver Version                                  2023.16.12.0.12_195853.xmain-hotfix
  Driver UUID                                     32332e35-322e-3238-3230-322e35320000
  Driver Version                                  23.52.28202.52
-----------------------------------------------------------------
Driver related package version:
ii  intel-fw-gpu                                    2024.04.6-293~22.04                     all          Firmware package for Intel integrated and discrete GPUs
ii  intel-i915-dkms                                 1.24.1.11.240117.14+i16-1               all          Out of tree i915 driver.
ii  intel-level-zero-gpu                            1.3.28202.52-821~22.04                  amd64        Intel(R) Graphics Compute Runtime for oneAPI Level Zero.
ii  level-zero-dev                                  1.16.15-821~22.04                       amd64        Intel(R) Graphics Compute Runtime for oneAPI Level Zero.
-----------------------------------------------------------------
igpu not detected
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
GPU0 Memory size=16G
-----------------------------------------------------------------
03:00.0 VGA compatible controller: Intel Corporation Device 56a0 (rev 08) (prog-if 00 [VGA controller])
        Subsystem: Device 1ef7:1307
        Flags: bus master, fast devsel, latency 0, IRQ 199
        Memory at 84000000 (64-bit, non-prefetchable) [size=16M]
        Memory at 4000000000 (64-bit, prefetchable) [size=16G]
        Expansion ROM at 85000000 [disabled] [size=2M]
        Capabilities: <access denied>
        Kernel driver in use: i915
        Kernel modules: i915
-----------------------------------------------------------------
```


* Windows Desktop equipped with i9 13900K Intel Core CPU and Intel(R) Arc(TM) A770 GPU example output:

```
Python 3.11.8
-----------------------------------------------------------------
transformers=4.37.2
-----------------------------------------------------------------
torch=2.1.0a0+cxx11.abi
-----------------------------------------------------------------
Name: ipex-llm
Version: 2.1.0b20240410
Summary: Large Language Model Develop Toolkit
Home-page: https://github.com/intel-analytics/BigDLy
Author: BigDL Authors
Author-email: bigdl-user-group@googlegroups.com
License: Apache License, Version 2.0
Location: C:\Users\arda\miniconda3\envs\ipex-llm-langchain-chatchat\Lib\site-packages
Requires:
Required-by:
-----------------------------------------------------------------
ipex=2.1.10+xpu
-----------------------------------------------------------------
Total Memory: 63.747 GB

Chip 0 Memory: 32 GB | Speed: 5600 MHz
Chip 1 Memory: 32 GB | Speed: 5600 MHz
-----------------------------------------------------------------
CPU Manufacturer: GenuineIntel
CPU MaxClockSpeed: 3000
CPU Name: 13th Gen Intel(R) Core(TM) i9-13900K
CPU NumberOfCores: 24
CPU NumberOfLogicalProcessors: 32
-----------------------------------------------------------------
GPU 0: Intel(R) Arc(TM) A770 Graphics    Driver Version: 31.0.101.5084
-----------------------------------------------------------------
System Information

Host Name:                 DESKTOP-ORSLCSS
OS Name:                   Microsoft Windows 11 Enterprise
OS Version:                10.0.22631 N/A Build 22631
OS Manufacturer:           Microsoft Corporation
OS Configuration:          Member Workstation
OS Build Type:             Multiprocessor Free
Registered Owner:          Intel User
Registered Organization:   Intel Corporation
Product ID:                00330-80000-00000-AA989
Original Install Date:     4/9/2024, 1:40:07 PM
System Boot Time:          4/12/2024, 12:50:50 PM
System Manufacturer:       HP
System Model:              HP EliteBook 840 G8 Notebook PC
System Type:               x64-based PC
Processor(s):              1 Processor(s) Installed.
                           [01]: Intel64 Family 6 Model 140 Stepping 1 GenuineIntel ~2995 Mhz
BIOS Version:              HP T37 Ver. 01.16.00, 1/18/2024
Windows Directory:         C:\WINDOWS
System Directory:          C:\WINDOWS\system32
Boot Device:               \Device\HarddiskVolume1
System Locale:             en-us;English (United States)
Input Locale:              en-us;English (United States)
Time Zone:                 (UTC+08:00) Beijing, Chongqing, Hong Kong, Urumqi
Total Physical Memory:     16,112 MB
Available Physical Memory: 3,723 MB
Virtual Memory: Max Size:  23,792 MB
Virtual Memory: Available: 9,035 MB
Virtual Memory: In Use:    14,757 MB
Page File Location(s):     C:\pagefile.sys
Domain:                    ccr.corp.intel.com
Logon Server:              \\PGSCCR601
Hotfix(s):                 5 Hotfix(s) Installed.
                           [01]: KB5034467
                           [02]: KB5027397
                           [03]: KB5036893
                           [04]: KB5035967
                           [05]: KB5037020
Network Card(s):           4 NIC(s) Installed.
                           [01]: Cisco AnyConnect Secure Mobility Client Virtual Miniport Adapter for Windows x64
                                 Connection Name: Ethernet 3
                                 Status:          Hardware not present
                           [02]: Intel(R) Wi-Fi 6 AX201 160MHz
                                 Connection Name: Wi-Fi
                                 DHCP Enabled:    Yes
                                 DHCP Server:     10.239.27.228
                                 IP address(es)
                                 [01]: 10.239.44.96
                                 [02]: fe80::95ba:2f4c:c5bf:74c
                           [03]: Bluetooth Device (Personal Area Network)
                                 Connection Name: Bluetooth Network Connection
                                 Status:          Media disconnected
                           [04]: PANGP Virtual Ethernet Adapter Secure
                                 Connection Name: Ethernet
                                 DHCP Enabled:    No
                                 IP address(es)
                                 [01]: 10.247.2.67
Hyper-V Requirements:      A hypervisor has been detected. Features required for Hyper-V will not be displayed.
-----------------------------------------------------------------
+-----------+--------------------------------------------------------------------------------------+
| Device ID | Device Information                                                                   |
+-----------+--------------------------------------------------------------------------------------+
| 0         | Device Name: Intel(R) Arc(TM) A770 Graphics                                          |
|           | Vendor Name: Intel(R) Corporation                                                    |
|           | UUID: 00000000-0000-0003-0000-000856a08086                                           |
|           | PCI BDF Address: 0000:03:00.0                                                        |
+-----------+--------------------------------------------------------------------------------------+
``` 
