# Bigdl-nano Pytorch Segmentation example on KITTI dataset

This example illustrates how to apply bigdl-nano optimizations on a semantic segmentation case based on pytorch-lightning framework. The basic semantic segmentation module is implemented with [Lightning](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/domain_templates/semantic_segmentation.py) and trained on [KITTI Semantic Segmentation Benchmark dataset](http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015). With all the optimizations provided by bigdl-nano, we could achieve 2.41x times faster than the official segmentation example in only a few lines of changes.


## Prepare the environment
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environment.
```
conda create -n nano python=3.7  # "nano" is conda environment name, you can use any name you like.
conda activate nano

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
Register an account in KITTI and access [KITTI Semantic Segmentation Benchmark dataset](http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015) for downloading the whole dataset(about 313MB). Unarchive it before running the example. 

## Run example
You can run this example with command line:

```bash
python semantic_segmentation_tmp.py --data_path data/ --use_ipex
```

**Options**
* `--data_path` The path to the unarchived dataset.Required.
* `--use_ipex` Whether we use ipex as accelerator for trainer. Default is False.
* `--num_processes` The number of processes in distributed training. Default: 1.

## Results

You can find the result for training as follows:
```
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
/opt/conda/envs/test37/lib/python3.7/site-packages/pytorch_lightning/trainer/configuration_validator.py:99: UserWarning: you passed in a val_dataloader but have no validation_step. Skipping val loop
  rank_zero_warn(f"you passed in a {loader_name} but have no {step_name}. Skipping {stage} loop")

  | Name | Type | Params
------------------------------
0 | net  | UNet | 31.0 M
------------------------------
31.0 M    Trainable params
0         Non-trainable params
31.0 M    Total params
124.179   Total estimated model params size (MB)
/opt/conda/envs/test37/lib/python3.7/site-packages/pytorch_lightning/trainer/data_loading.py:106: UserWarning: The dataloader, train dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 96 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
  f"The dataloader, {name}, does not have many workers which may be a bottleneck."
/opt/conda/envs/test37/lib/python3.7/site-packages/pytorch_lightning/trainer/data_loading.py:323: UserWarning: The number of training samples (10) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.
  f"The number of training samples ({self.num_training_batches}) is smaller than the logging interval"
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [02:14<00:00, 12.26s/it, loss=2.35, v_num=64]
```
