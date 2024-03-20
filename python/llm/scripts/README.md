#  Utility Scripts


## Env-Check

The **Env-Check** scripts  ([env-check.sh](./env-chec.sh), [env-check.bat](./env-chec.bat)) are designed to verify your `bigdl-llm` installation and runtime environment. These scripts can help you ensure your environment is correctly set up for optimal performance. You can include the script's output when reporting issues on [BigDL Github Issues](https://github.com/intel-analytics/BigDL/issues) for easier troubleshooting.

> Note: These scripts verify python installation, check for necessary packages and environmental variables, assess hardware or operating system compatibility, and identify any XPU-related issues. 

### Install extra dependency

* On Linux, the script uses a tool named `xpu-smi`. It is a convinent tool the monitor ths status of the GPUs. If you're runing LLMs on GPUs on Linux, we recommend installing `xpu-smi`. Run below command to install:
```
sudo apt install xpu-smi
```
* On Windows, you can ignore the sections in `xpu-smi.exe` if you didn't install it. You can always use **Windows Task Manager** to monitor the status of GPUs on Windows.   
  
### Usage

* After installing `bigdl-llm`, open a terminal (on Linux) or **Anaconda Prompt** (on Windows), and activate the conda environment you have created for running `bigdl-llm`: 
  ```
  conda activate llm
  ```
  > If you do not know how to install `bigdl-llm`, refer to [BigDL-LLM installation](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install.html) for more details.
*  Within the activated python environment, run below command:
    *  On Linux
        ```bash
        bash env-check.sh
        ```
    * On Windows,
        ```bash
        env-check.bat
        ```

### Sample outputs

* An example output on a Linux Desktop equipped with i9 13-Gen Intel Core CPU and Intel(R) Arc(TM) A770 GPU looks like below. 

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
USER_ZDOTDIR=/home/user
COLORTERM=truecolor
LESS=-R
TERM_PROGRAM_VERSION=1.86.2
ONEAPI_ROOT=/opt/intel/oneapi
CONDA_EXE=/home/user/anaconda3/bin/conda
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
