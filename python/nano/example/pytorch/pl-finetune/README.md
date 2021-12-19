# Bigdl-nano Fine-tune example on Cat vs. Dog dataset

This example illustrates how to apply bigdl-nano optimizations on a fine-tuning case based on pytorch-lightning framework. For the sake of this example, we train the proposed network(by default, a ResNet50 is used) on the [cats and dogs dataset](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip), which consists both [frozen and unfrozen stages](https://github.com/PyTorchLightning/pytorch-lightning/blob/495812878dfe2e31ec2143c071127990afbb082b/pl_examples/domain_templates/computer_vision_fine_tuning.py#L21-L35). With all the optimizations provided by bigdl-nano, we could achieve 5x times faster than the official PL finetune example in only 3 lines of changes.


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

## Run example
You can run this example with command line:

```bash
python pl-finetune.py
```

**Options**
* `--trainer.num_processes` The number of processes in distributed training.Default: 1.
* `--trainer.use_ipex` Whether we use ipex as accelerator for trainer. Default is False.
* `--trainer.distributed_backend` The way distributing model and data, `spawn` and `ray` are available. Default is `spawn`.
* `--model.backbone` Name (as in ``torchvision.models``) of the feature extractor. Default is `resnet50`.

## Results

You can find the result for training as follows:
```
Global seed set to 1234
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
Downloading https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip to data/cats_and_dogs_filtered.zip
68606976it [00:07, 9476966.43it/s]                                                                                                                                                                                                                              
Extracting data/cats_and_dogs_filtered.zip to data
The model will start training with only 6 trainable parameters out of 165.
/opt/conda/envs/test37/lib/python3.7/site-packages/pytorch_lightning/trainer/data_loading.py:106: UserWarning: The dataloader, train dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 96 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
  f"The dataloader, {name}, does not have many workers which may be a bottleneck."
/opt/conda/envs/test37/lib/python3.7/site-packages/pytorch_lightning/trainer/data_loading.py:106: UserWarning: The dataloader, val dataloader 0, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 96 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
  f"The dataloader, {name}, does not have many workers which may be a bottleneck."
Epoch 0:  75%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                           | 280/375 [01:01<00:20,  4.57it/s, loss=0.116, v_num=4, train_acc=0.875]
Validating:  24%|█████████████████████████████████████████████████▍                                                                                                                                                            | 30/125 [00:07<00:24,  3.95it/s]
```
