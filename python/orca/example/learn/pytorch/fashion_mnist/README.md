# PyTorch Fashion-MNIST example with Tensorboard visualization
We demonstrate how to easily show the graphical results of running synchronous distributed PyTorch training using PyTorch Estimator of Project Orca in Analytics Zoo. We use a simple convolutional nueral network model to train on fashion-MNIST dataset. See [here](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html) for the original single-node version of this example provided by PyTorch. We provide three distributed PyTorch training backends for this example, namely "bigdl", "torch_distributed" and "spark". You can run with either backend as you wish.

## Prepare the environment

We recommend you to use Anaconda to prepare the environment, especially if you want to run on a yarn cluster:

```
conda create -n bigdl python=3.7  # "bigdl" is conda environment name, you can use any name you like.
conda activate bigdl
pip install torch
pip install torchvision
pip install matplotlib
pip install tensorboard

# For bigdl backend
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

## Run on local after pip install

The default backend is `bigdl`.

```
python fashion_mnist.py
```

You can run with `torch_distributed` backend via:

```
python fashion_mnist.py --backend torch_distributed
```

To see the result figures after it finishes:

```
tensorboard --logdir=runs
```

Then open `https://localhost:6006`.

You can run with `spark` backend via:

```
python fashion_mnist.py --backend spark
```

## Run on yarn cluster for yarn-client mode after pip install

```
export HADOOP_CONF_DIR=the directory of the hadoop and yarn configurations
python fashion_mnist.py --cluster_mode yarn
```

Then open `https://localhost:6006` on the local client machine to see the result figures.

The default backend is `bigdl`. You can also run with `torch_distributed` or `spark` by specifying the backend.

## Results

**For "bigdl" backend**

You can find the logs for training as follows:

```
22-02-08 16:21:13 INFO  DistriOptimizer$:430 - [Epoch 1 55600/60000][Iteration 13900][Wall Clock 454.145287871s] Trained 4.0 records in 0.019744969 seconds. Throughput is 202.58324 records/second. Loss is 0.9015253.
```

Final test results will be printed at the end:

```
Accuracy of the network on the test images: {'Top1Accuracy': 0.8704000115394592}
```

**For "torch_distributed" and "spark" backend**

You can find the results of training and validation as follows:

```
# torch_distributed backend
Train stats: [{'num_samples': 60000, 'epoch': 1, 'batch_count': 938, 'train_loss': 1.7453054378827413, 'last_train_loss': 0.5022475123405457}, {'num_samples': 60000, 'epoch': 2, 'batch_count': 938, 'train_loss': 0.7086501805464427, 'last_train_loss': 0.32789549231529236}]

Validation stats: {'num_samples': 10000, 'Accuracy': tensor(0.7557), 'val_loss': 0.6404335161209106}

# spark backend
Train stats: [{'num_samples': 60000, 'epoch': 1, 'batch_count': 938, 'train_loss': 1.6330145305315653, 'last_train_loss': 0.5567581057548523}, {'num_samples': 60000, 'epoch': 2, 'batch_count': 938, 'train_loss': 0.7078083839098612, 'last_train_loss': 0.31840038299560547}]

Validation stats: {'num_samples': 10000, 'Accuracy': tensor(0.7633), 'val_loss': 0.630250259399414}
```
