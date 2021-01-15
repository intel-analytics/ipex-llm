# PyTorch Cifar10 example
We demonstrate how to easily run synchronous distributed PyTorch training using PyTorch Estimator of Project Orca in Analytics Zoo. We use a simple convolutional nueral network model to train on Cifar10 dataset, which is a dataset for image classification. See [here](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) for the original single-node version of this example provided by PyTorch.

## Prepare the environment

We recommend you to use Anaconda to prepare the environment, especially if you want to run on a yarn cluster:

```
conda create -n zoo python=3.7  # "zoo" is conda environment name, you can use any name you like.
conda activate zoo
pip install analytics-zoo  # 0.9.0 or above
pip install torch
pip install torchvision
```

## Run on local after pip install

```
python cifar10.py
```

## Run on yarn cluster for yarn-client mode after pip install

```
export HADOOP_CONF_DIR=the directory of the hadoop and yarn configurations
python cifar10.py --cluster_mode yarn-client
```

## Results

You can find the logs for training as follows:
```
2020-12-03 15:25:30 INFO  DistriOptimizer$:426 - [Epoch 2 47680/50000][Iteration 24420][Wall Clock 497.634203315s] Trained 4 records in 0.022554577 seconds. Throughput is 177.3476 records/second. Loss is 0.82751834.
```

Final test results will be printed at the end:
```
Accuracy of the network on the 10000 test images: 0.541100025177002 
```
