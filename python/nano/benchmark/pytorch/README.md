# Nano Pytorch cat-vs-dog Example
This computer vision example illustrates how one could use BigDL nano to easily train 
a Resnet50 model to do cat and dog classfication. 

## Prepare the environment
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environment.

```
conda create -n nano python=3.7  # "nano" is conda environment name, you can use any name you like.
conda activate nano
pip install jsonargparse[signatures]

pip install bigdl-nano[pytorch]
```
Initialize environment variables with script `bigdl-nano-init` installed with bigdl-nano.
```
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
## Prepare Dataset
By default the dataset will be auto-downloaded.
You could access [cats and dogs dataset](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip) for a view of the whole dataset.

## Performance Test
Test performance on pytorch cat-vs-dog example with command line:

```bash
python pytorch-cat-vs-dog.py
```
**Options**
* `--batch_size` training batch size.Default: 32.
* `--epochs` training epoch number for performance test.Default: 2.
* `--root_dir` path to cat vs dog dataset which should have two folders `cat` and `dog`, each containing cat and dog pictures. Default is None, i.e. download dataset before training.
* `--freeze` if force finetune freezed, None to test both cases, True/False to test corresponding case
* `--remove_data` if to remove dataset after performance test. Default is true, i.e. remove after test
* `--output_to_csv` if output performance test result to csv file. Default: True
* `--csv_path` output performance test result to csv file

## Performance Result
### ResNet 50
metric： throughput (img/s) = imgs * epochs / time_interval
