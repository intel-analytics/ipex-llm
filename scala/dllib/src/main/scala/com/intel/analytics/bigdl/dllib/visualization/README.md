# Visualization via Tensorboard

> This wheel distribution is provided by https://github.com/dmlc/tensorboard

> You can find the wheel repository at https://pypi.python.org/pypi/tensorboard

Please follow the instructions below to install TensorBoard; it has been tested on both Ubuntu and Mac OS. Please refer to the [Know Issues](https://github.com/122689305/BigDL/tree/readme/spark/dl/src/main/scala/com/intel/analytics/bigdl/visualization#known-issues)  section for possible errors.

## Requirement

Python verison: 2.7, 3.4, 3.5, 3.6

Pip version >= 9.0.1

## Installation

### Python 2
```pip install tensorboard==1.0.0a4```
### Python 3
```pip3 install tensorboard==1.0.0a4```

## Known Issues

> #### 1. Issue: No compatible version of tensorboard

Solutions
*  [Update](https://pip.pypa.io/en/stable/installing/) your pip version to the latest: https://pip.pypa.io/en/stable/installing/
*  Check whether your python support wide unicode if you use python 2.7 
```
python -c 'import sys;print(sys.maxunicode)'
```
　　It should output `1114111`

> #### 2. RuntimeError: module compiled against API version 0xa but this version of numpy is 0x9

　　Check your python library path (sys.path) to see whether it includes numpy module

> #### 3. RuntimeError: Cannot load some specific libraries, like '_pywrap_tensorflow.so'. 

　　Set your 'PATH' environment variable so that `$ which python` outputs the path of your python that has installed tensorboard.
