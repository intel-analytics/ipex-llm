# Orca PyTorch Super Resolution example on BSDS300 dataset

We demonstrate how to easily run synchronous distributed Pytorch training using Pytorch Estimator of Project Orca in Analytics Zoo. This is an example using the efficient sub-pixel convolution layer to train on [BSDS3000 dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), using crops from the 200 training images, and evaluating on crops of the 100 test images. See [here](https://github.com/pytorch/examples/tree/master/super_resolution) for the original single-node version of this example provided by Pytorch. We provide three distributed PyTorch training backends for this example, namely "bigdl", "torch_distributed" and "spark". You can run with either backend as you wish.

## Prepare the environment
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environment, especially if you want to run on a yarn cluster (yarn-client mode only).
```
conda create -n bigdl python=3.7  # "bigdl" is conda environment name, you can use any name you like.
conda activate bigdl
pip install pillow
conda install pytorch torchvision cpuonly -c pytorch  # command for linux
conda install pytorch torchvision -c pytorch  # command for macOS

# For bigdl backend:
pip install bigdl-orca
pip install jep==3.9.0
pip install six cloudpickle

# For torch_distributed backend:
pip install bigdl-orca[ray]
pip install tqdm  # progress bar

# For spark backend
pip install bigdl-orca
pip install tqdm  # progress bar
```

## Prepare Dataset
By default dataset will be auto-downloaded on local mode and yarn-client mode.
If your yarn nodes can't access internet, run the `prepare_dataset.sh` to prepare dataset automatically.
```
bash prepare_dataset.sh
```
After running the script, you will see  **dataset (for local mode use)** and archive **dataset.zip (for yarn mode use)** in the current directory.

## Run example
You can run this example on local mode (default) and yarn-client mode.

- Run with Spark Local mode:
```bash
python super_resolution.py --cluster_mode local
```

- Run with Yarn-Client mode:
```bash
python super_resolution.py --cluster_mode yarn
```

You can run this example with bigdl backend (default) or torch_distributed backend.

- Run with bigdl backend:
```bash
python super_resolution.py --backend bigdl
```

- Run with torch_distributed backend:
```bash
python super_resolution.py --backend torch_distributed
```

- Run with spark backend:
```bash
python super_resolution.py --backend spark
```

**Options**
* `--upscale_factor` The upscale factor of super resolution. Default is 3.
* `--batch_size` The number of samples per gradient update. Default is 64.
* `--test_batch_size` The number of samples per batch validate. Default is 10.
* `--lr` Learning Rate. Default is 0.01.
* `--epochs` The number of epochs to train for. Default is 2.
* `--cluster_mode` The mode of spark cluster. Either "local" or "yarn". Default is "local".
* `--backend` The backend of PyTorch Estimator. Either "bigdl", "torch_distributed" or "spark. Default is "bigdl".
* `--data_dir` The path of datesets. Default is "./dataset".

## Results

**For "bigdl" backend**

You can find the result for training as follows:
```
22-02-08 16:53:41 INFO  DistriOptimizer$:430 - [Epoch 1 256/256][Iteration 4][Wall Clock 24.509990412s] Trained 64.0 records in 0.843214699 seconds. Throughput is 75.90001 records/second. Loss is 0.1878621.
```
You can find the result for validation as follows:
```
22-02-08 16:54:11 INFO  DistriOptimizer$:1512 - MSE is (Loss: 0.094448544, count: 2, Average Loss: 0.047224272)
===> Validation Complete: Avg. PSNR: 13.2583 dB, Avg. Loss: 0.0472
```

**For "torch_distributed" backend**
You can find the result for training as follows:
```
===> Epoch 2 Complete: Avg. Loss: 0.1025
```
You can find the result for validation as follows:
```
===> Validation Complete: Avg. PSNR: 12.8203 dB, Avg. Loss: 0.0522
```

**For "spark" backend**

You can find the result for training as follows:
```
===> Epoch 2 Complete: Avg. Loss: 0.1025
```
You can find the result for validation as follows:
```
===> Validation Complete: Avg. PSNR: 12.8203 dB, Avg. Loss: 0.0522
```
