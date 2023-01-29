# Installation


## Scala

Refer to [BigDl Install guide for Scala](../../UserGuide/scala.md).


## Python


### Install a Stable Release

Run below command to install _bigdl-dllib_.

```bash
conda create -n my_env python=3.7
conda activate my_env
pip install bigdl-dllib
```

### Install Nightly build version

You can install the latest nightly build of bigdl-dllib as follows:
```bash
pip install --pre --upgrade bigdl-dllib
```

### Verify your install

You may verify if the installation is successful using the interactive Python shell as follows:

* Type `python` in the command line to start a REPL.
* Try to run the example code below to verify the installation:

  ```python
  from bigdl.dllib.utils.nncontext import *

  sc = init_nncontext()  # Initiation of bigdl-dllib on the underlying cluster.
  ```

