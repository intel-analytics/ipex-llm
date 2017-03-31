# Tensorboard Installation Guide


---

> This wheel distribution is lauched by this group:https://github.com/dmlc/tensorboard

>You can find the wheel repository at https://pypi.python.org/pypi/tensorboard


Simply, with one pip command you can install it in Ubuntu. But it is very likely that you will run into problems while installing. If you meet any error, please check the instructions below.

If you install it with Mac OS, similar in ubuntu, it is easy to install with pip. Notice that the wheels for mac do not require CPython to support wide unicode. If you meet any error, read the instructions below carefully. You may find your own solutions from these information. 

### Python 2
```pip install tensorboard==1.0.0a4```
### Python 3
```pip3 install tensorboard==1.0.0a4```

## Troubleshooting Section
----------

> #### 1 . Wrong Installation Path

Notice that pip is only a script to launch corresponding python. Make sure the 'pip' that you use will install the package to the python directory that you want.
Run these two commands, and see whether both results point to the same python binary execution path.
```
head -n 1 $(which pip) 
which python
```

> #### 2 . No compatible version of tensorboard
Skipping ... because it is not compatible with this Python 

To understand why this kind of problem occurs, first you have to understand the naming rule of wheel. It's `{distribution}-{version}(-{build tag})?-{python tag}-{abi tag}-{platform tag}.whl`. Take this tensorboard for example: `tensorboard-1.0.0a4-cp27-cp27mu-manylinux1_x86_64.whl`. The python tag is cp27, which means a python 2.7.* with CPython as implementation. The abi tag is cp27mu, which means it requires the CPython compiled with `--with-pymalloc` and `--with-wide-unicode`. If you installed pip, you can check whether your python support it or not by running this command:
```
python -c 'from pip import pep425tags; print(pep425tags.supported_tags)' 
```
Then you may get something like this:
```
[('cp27', 'cp27mu', 'manylinux1_x86_64'), ('cp27', 'cp27mu', 'linux_x86_64'), ('cp27', 'none', 'manylinux1_x86_64'), ('cp27', 'none', 'linux_x86_64'), ('py2', 'none', 'manylinux1_x86_64'), ('py2', 'none', 'linux_x86_64'), ('cp27', 'none', 'any'), ('cp2', 'none', 'any'), ('py27', 'none', 'any'), ('py2', 'none', 'any'), ('py26', 'none', 'any'), ('py25', 'none', 'any'), ('py24', 'none', 'any'), ('py23', 'none', 'any'), ('py22', 'none', 'any'), ('py21', 'none', 'any'), ('py20', 'none', 'any')]
```
Notice that the first entry of tag tupple means that this python can install the above wheel.

**But you may get this instead**
```
[('cp27', 'none', 'linux_x86_64'), ('cp27', 'none', 'any'), ('cp2', 'none', 'any'), ('cp26', 'none', 'any'), ('cp25', 'none', 'any'), ('cp24', 'none', 'any'), ('cp23', 'none', 'any'), ('cp22', 'none', 'any'), ('cp21', 'none', 'any'), ('cp20', 'none', 'any'), ('py27', 'none', 'any'), ('py2', 'none', 'any'), ('py26', 'none', 'any'), ('py25', 'none', 'any'), ('py24', 'none', 'any'), ('py23', 'none', 'any'), ('py22', 'none', 'any'), ('py21', 'none', 'any'), ('py20', 'none', 'any')]
```
**or the first entry of tag tupple is ('cp27', 'cp27m', 'manylinux1_x86_64'), without --with-wide-unicode support**

**As to these conditons, I'll explain to you**

 - *The first conditon is caused by low pip version.*

Check your pip version: `$ pip -V`. It should be higher than 9. Otherwise, upgrade it. If you install it by `$ sudo apt install python-pip`, after `$sudo apt update` `$ sudo apt install python-pip --upgrade`, you may get this response `python-pip is already the newest version`. Check the pip version, if it is not the latest yet, run `$ pip install pip --upgrade` or `$ pip install pip==9.0.1`. If it reports `Not uninstalling pip at /usr/lib/python2.7/dist-packages, owned by OS`, then you have to remove your pip intsalled by `$ sudo apt install python-pip`. It is because that `apt` and `pip` are different package management tool, and `apt` will by default installs staff into `'/usr/'` while `pip` will by default installs staff into `/usr/local`. 
**[Solution]** To remove `pip` installed by `apt` and install latest `pip`, do 
```
sudo apt remove pip
rm /usr/local/lib/python2.7/dist-packages/pip* -rf
rm /usr/local/bin/pip*
```
and then 
```
wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py
```
   
- *The second condition is caused by your python compilation*

    **Please read this section after `pip` is upgraded to the latest.**
	If you install tensorboard with `python 2`, which version is `>= 2.7.10`, then this error may occur. This is because in `python < 2.7.10`, it is built in `--enable-unicode=ucs4`, but in python `>= 2.7.10`, it is `--enable-unicode=ucs2`. `ucs4` means `4 Bytes` unicode, while `ucs2` means `2 Bytes` unicode.Take this tensorboard wheel for python 2.7 as example: `tensorboard-1.0.0a4-cp27-cp27mu-manylinux1_x86_64.whl`. Notice that `cp27mu` means it requires python with CPython implementation which enables `pymalloc` and `wide-unicode`. `wide-unicode` means `4 Bytes` unicode. You may check your python implementation by running 
```
python -c 'import platform;print(platform.python_implementation())'
```
*(substitute 'python' with 'python3' if you want use python 3)*
also you can check whether your CPython supports `ucs2` or `ucs4` by running
```
python -c 'import sys;print(sys.maxunicode)'
```
If it is `1114111`, then it's `ucs4`. If it is `65535`, then it is `ucs2`.
	
**So if you find that your unicode setting is different from the wheel requirement, then you may re-compile your python.**
```
cd /path/to/your/python/source
./configure --enbale-unicode=ucs4 --prefix=/dir/to/install
make clean
make
make install
```
	 
> #### 3. RuntimeError: module compiled against API version 0xa but this version of numpy is 0x9
ImportError: numpy.core.multiarray failed to import

This is caused by the low version of `numpy`. But since you have already updated it by `pip` while installing `tensorboard`, how can it comes? It is because that you have multiple versions of `numpy`. Check your python `sys.path`, to see the priority of the python package searching path:
```
python -c 'import sys;print(sys.path)'
```
If you get something like this:
```
['', '/usr/lib/python2.7/dist-packages', '/usr/lib/python2.7', '/usr/lib/python2.7/plat-x86_64-linux-gnu', '/usr/lib/python2.7/lib-tk', '/usr/lib/python2.7/lib-old', '/usr/lib/python2.7/lib-dynload', '/usr/local/lib/python2.7/dist-packages']
```
It means that your python will first check system installed packages (like form `apt`) (begin from `/usr`) and then go to custom installed packages (like from `pip`) (begin from `/usr/local`). One solution is to define your environment variable `PYTHONPATH` to `'/usr/local/lib/python2.7/dist-packages'`, because `PYTHONPATH` is given the highest priority second to `''`. But I don't recommend this method, since the packages installed there are from `apt`. It is better to manage python packages with `pip`. So another solution is to remove the `numpy` installed by `apt`, and re-install it with latest `pip`:
```
sudo apt remove python-numpy
pip uninstall numpy # please repeat it several times, in case that multiple numpy versions have been installed
pip install numpy
```


> #### 4. RuntimeError: Cannot load some specific libraries, like '_pywrap_tensorflow.so'. 

This is caused by the wrong setting of python binary path in the launch script.
You can find an executable script `tensorboard` in `your/python/path/bin/`. Check it, then you can find that this script uses a default `PYTHON_BINARY` path as its executable python path,  which is 
`PYTHON_BINARY = '/home/travis/virtualenv/python3.6.0/bin/python'`
Here travis may be the author of this `tensorboard` wheel.  Generally speaking, our python path is not there. In such case, this script provides another way to find an executable python path. It will automatically search for an executable python whose name is `python` in `$PATH`. So for example, if in your platform you use `python3` command to launch your Python 3.6.0, and you install this tensorboard to your Python 3.6.0, then this script may mis-launch the tensorboard with  Python 2.7, whose command is `python`.

So the solution is:
```
sed -i '/^PYTHON_BINARY/c PYTHON_BINARY = "/path/to/your/python binary"' $(which tensorboard)
```
