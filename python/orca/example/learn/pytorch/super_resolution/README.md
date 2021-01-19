# Orca PyTorch Super Resolution example on BSDS300 dataset

We demonstrate how to easily run synchronous distributed Pytorch training using Pytorch Estimator of Project Orca in Analytics Zoo. This is an example using the efficient sub-pixel convolution layer to train on [BSDS3000 dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), using crops from the 200 training images, and evaluating on crops of the 100 test images. See [here](https://github.com/pytorch/examples/tree/master/super_resolution) for the original single-node version of this example provided by Pytorch.

## Prepare the environment
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environment, especially if you want to run on a yarn cluster (yarn-client mode only).
```
conda create -n zoo python=3.7  # "zoo" is conda environment name, you can use any name you like.
conda activate zoo
pip install analytics-zoo[ray]  # 0.9.0 or above
pip install pillow
conda install pytorch torchvision cpuonly -c pytorch  # command for linux
conda install pytorch torchvision -c pytorch  # command for macOS
```

## Prepare Dataset
By default dataset will be auto-downloaded on local mode and yarn-client mode.
If your yarn nodes can't access internet, run the `prepare_dataset.sh` to prepare dataset automatically.
```
bash prepare_dataset.sh
```
After running the script, you will see  **dataset (for local mode use)** and archive **dataset.zip (for yarn mode use)** in the current directory.

## Run example
You can run this example on local mode and yarn-client mode.

- Run with Spark Local mode:
```bash
python super_resolution.py --cluster_mode local
```

- Run with Yarn-Client mode:
```bash
python super_resolution.py --cluster_mode yarn
```

**Options**
* `--upscale_factor` The upscale factor of super resolution. Default is 3.
* `--batch_size` The number of samples per gradient update. Default is 64.
* `--test_batch_size` The number of samples per batch validate. Default is 10.
* `--lr` Learning Rate. Default is 0.01.
* `--epochs` The number of epochs to train for. Default is 2.
* `--cluster_mode` The mode of spark cluster. Either "local" or "yarn". Default is "local".

## Results
You can find the result for training as follows:
```
===> Epoch 1 Complete: Avg. Loss: 9.2605
```
You can find the result for validation as follows:
```
===> Validation Complete: Avg. PSNR: 7.2006 dB, Avg. Loss: 0.1905
```