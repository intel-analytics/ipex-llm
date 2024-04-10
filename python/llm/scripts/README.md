#  Utility Scripts


## Env-Check

The **Env-Check** scripts  ([env-check.sh](./env-chec.sh), [env-check.bat](./env-chec.bat)) are designed to verify your `ipex-llm` installation and runtime environment. These scripts can help you ensure your environment is correctly set up for optimal performance. You can include the script's output when reporting issues on [IPEX Github Issues](https://github.com/intel-analytics/ipex-llm/issues) for easier troubleshooting.

> Note: These scripts verify python installation, check for necessary packages and environmental variables, assess hardware or operating system compatibility, and identify any XPU-related issues. 

### Install extra dependency

* On Linux, the script uses a tool named `xpu-smi`. It is a convinent tool the monitor ths status of the GPUs. If you're runing LLMs on GPUs on Linux, we recommend installing `xpu-smi`. Run below command to install:
```
sudo apt install xpu-smi
```
* On Windows, you can ignore the sections in `xpu-smi.exe` if you didn't install it. You can always use **Windows Task Manager** to monitor the status of GPUs on Windows.
  
### Usage

* After installing `ipex-llm`, open a terminal (on Linux) or **Anaconda Prompt** (on Windows), and activate the conda environment you have created for running `ipex-llm`: 
  ```
  conda activate llm
  ```
  > If you do not know how to install `ipex-llm`, refer to [IPEX-LLM installation](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Overview/install.html) for more details.
*  Within the activated python environment, run below command:
    *  On Linux
      * install clinfo `sudo apt install clinfo`

      * activate oneapi
        For IPEX 2.0, run this `source /home/arda/intel/oneapi/setvars.sh`

        For IPEX 2.0, run this `source /opt/intel/oneapi/setvars.sh`

      * run the env check
        ```bash
        bash env-check.sh
        ```
    * On Windows,
        ```bash
        env-check.bat
        ```

### Sample outputs

* An example output on a Linux Desktop equipped with i9 13900K Intel Core CPU and Intel(R) Arc(TM) A770 GPU looks like below.

```
-----------------------------------------------------------------
PYTHON_VERSION=3.9.18
-----------------------------------------------------------------
transformers=4.31.0
-----------------------------------------------------------------
torch=2.1.0a0+cxx11.abi
-----------------------------------------------------------------
ipex-llm Version: 2.1.0b20240325
-----------------------------------------------------------------
ipex=2.1.10+xpu
-----------------------------------------------------------------
CPU Information: 
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Address sizes:                      46 bits physical, 48 bits virtual
Byte Order:                         Little Endian
CPU(s):                             24
On-line CPU(s) list:                0-23
Vendor ID:                          GenuineIntel
Model name:                         12th Gen Intel(R) Core(TM) i9-12900K
CPU family:                         6
Model:                              151
Thread(s) per core:                 2
Core(s) per socket:                 16
Socket(s):                          1
Stepping:                           2
CPU max MHz:                        5200.0000
CPU min MHz:                        800.0000
BogoMIPS:                           6374.40
-----------------------------------------------------------------
Total CPU Memory: 62.5343 GB
Memory Type: DDR4 
-----------------------------------------------------------------
Operating System: 
Ubuntu 22.04.3 LTS \n \l

-----------------------------------------------------------------
Linux arda-arc05 6.2.0-39-generic #40~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Thu Nov 16 10:53:04 UTC 2 x86_64 x86_64 x86_64 GNU/Linux
-----------------------------------------------------------------
CLI:
    Version: 1.2.22.20231025
    Build ID: 00000000

Service:
    Version: 1.2.22.20231025
    Build ID: 00000000
    Level Zero Version: 1.14.0
-----------------------------------------------------------------
  Driver Version                                  2023.16.10.0.17_160000
  Driver Version                                  2023.16.10.0.17_160000
  Driver UUID                                     32332e33-352e-3237-3139-312e34320000
  Driver Version                                  23.35.27191.42
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
        Flags: bus master, fast devsel, latency 0, IRQ 175
        Memory at 84000000 (64-bit, non-prefetchable) [size=16M]
        Memory at 4000000000 (64-bit, prefetchable) [size=16G]
        Expansion ROM at 85000000 [disabled] [size=2M]
        Capabilities: <access denied>
        Kernel driver in use: i915
        Kernel modules: i915
-----------------------------------------------------------------
```
