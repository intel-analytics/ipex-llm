# Visualization via Tensorboard


---

> This wheel distribution is provided by https://github.com/dmlc/tensorboard

> You can find the wheel repository at https://pypi.python.org/pypi/tensorboard


Simply, with one pip command you can install it in Ununtu. If you meet any error, please check the known issues below.

If you install it with Mac OS, similar in ubuntu, it is easy to install with pip.

## Requirement

Python verison: 2.7, 3.4, 3.5, 3.6

Pip version >= 9.0.1

## Installation

### Python 2
```pip install tensorboard==1.0.0a4```
### Python 3
```pip3 install tensorboard==1.0.0a4```

## Known Issues
----------
> #### 1. Issue: No compatible version of tensorboard

Solutions
1.  [Update](https://pip.pypa.io/en/stable/installing/) your pip version to the latest: https://pip.pypa.io/en/stable/installing/
2.  Check whether your python support wide unicode if you use python 2.7 
```
python -c 'import sys;print(sys.maxunicode)'
```
It should output `1114111`

> #### 2. RuntimeError: module compiled against API version 0xa but this version of numpy is 0x9

1. Check your python library path (sys.path) to see whether it includes numpy module

> #### 3. RuntimeError: Cannot load some specific libraries, like '_pywrap_tensorflow.so'. 

1. Set your 'PATH' environment variable so that `$ which python` outputs the path of your python that has installed tensorboard.
