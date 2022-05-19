---
## Bigdl-nano Resnet example on CIFAR10 dataset
This example illustrates how to apply bigdl-nano optimizations on a image recognition case based on pytorch-lightning framework. The basic image recognition module is implemented with Lightning and trained on [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) image recognition Benchmark dataset.
### Prepare the environment
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environment.
```bash
conda create -n nano python==3.7 # "nano" is conda environment name, you can use any name you like.
conda activate nano
```
#### Bigdl-nano
```bash
pip install bigdl-nano[pytorch]
```
Initialize environment variables with script `bigdl-nano-init` installed with bigdl-nano.
```bash
source bigdl-nano-init
```
You may find environment variables set like follows:
```
Setting OMP_NUM_THREADS...
Setting OMP_NUM_THREADS specified for pytorch...
Setting KMP_AFFINITY...
Setting KMP_BLOCKTIME...
Setting MALLOC_CONF...
+++++ Env Variables +++++
LD_PRELOAD=./../lib/libjemalloc.so
MALLOC_CONF=oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1
OMP_NUM_THREADS=48
KMP_AFFINITY=granularity=fine,compact,1,0
KMP_BLOCKTIME=1
TF_ENABLE_ONEDNN_OPTS=
+++++++++++++++++++++++++
Complete.
```
#### Jupyter Notebook
Install jupyter notebook
```bash
pip install jupyter
```
Configure jupyter notebook
- generate the jupyter configuration file
    ```ipython```
    ```bash
    In [1]: from notebook.auth import passwd
    In [2]: passwd()
    Enter password: ******
    Verify password: ******
    Out [2]: 'sha1:...'
    ```
    Save the generated key
- Modify the configuration file
   ```bash
   vim ~/.jupyter/jupyter_notebook_config.py
   ```
- Make the following changes in the file or add them directly at the end of the file
  ```bash
  c.NotebookApp.allow_remote_access = True 
  c.NotebookApp.ip='*'
  c.NotebookApp.password = u'sha:..' # The key previously generated
  c.NotebookApp.open_browser = False 
  c.NotebookApp.port =8888 # Specify any port
  ```
Start jupyter notebook
  ```bash
  jupyter notebook
  ```