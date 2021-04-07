# PyTorch Fashion-MNIST example with Tensorboard visualization
We demonstrate how to easily show the graphical results of running synchronous distributed PyTorch training using PyTorch Estimator of Project Orca in Analytics Zoo. We use a simple convolutional nueral network model to train on fashion-MNIST dataset. See [here](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html) for the original single-node version of this example provided by PyTorch. We provide two distributed PyTorch training backends for this example, namely "bigdl" and "torch_distributed". You can run with either backend as you wish.

## Prepare the environment

We recommend you to use Anaconda to prepare the environment, especially if you want to run on a yarn cluster:

```
conda create -n zoo python=3.7  # "zoo" is conda environment name, you can use any name you like.
conda activate zoo
pip install torch
pip install torchvision
pip install matplotlib

# For bigdl backend
pip install analytics-zoo  # 0.10.0.dev3 or above
pip install jep==3.9.0
pip install six cloudpickle

# For torch_distributed backend:
pip install analytics-zoo[ray]  # 0.10.0.dev3 or above
```

## Run on local after pip install

The default backend is `bigdl`.

```
python fashion_mnist.py
```

You can also run with `torch_distributed` backend via:

```
python fashion_mnist.py --backend torch_distributed
```

To see the result figures after it finishes:

```
tensorboard --logdir=runs
```

Then open `https://localhost:6006`.

## Run on yarn cluster for yarn-client mode after pip install

```
export HADOOP_CONF_DIR=the directory of the hadoop and yarn configurations
python fashion_mnist.py --cluster_mode yarn
```

Then open `https://localhost:6006` on the local client machine to see the result figures.

The default backend is `bigdl`. You can also run with `torch_distributed` by specifying the backend.

## Results

**For "bigdl" backend**

You can find the logs for training as follows:

```
2021-03-24 14:33:36 INFO  DistriOptimizer$:427 - [Epoch 4 52876/60000][Iteration 58219][Wall Clock 1230.685682812s] Trained 4.0 records in 0.016452279 seconds. Throughput is 243.12741 records/second. Loss is 0.0136261955.
```

Final test results will be printed at the end:

```
2021-03-24 14:39:43 INFO  DistriOptimizer$:1759 - Top1Accuracy is Accuracy(correct: 8851, count: 10000, accuracy: 0.8851)
```

**For "torch_distributed" backend**

You can find the results of training and validation as follows:

```
Train stats: [{'num_samples': 60000, 'epoch': 1, 'batch_count': 15000, 'train_loss': 0.6387080065780457, 'last_train_loss': 0.17801283299922943}, {'num_samples': 60000, 'epoch': 2, 'batch_count': 15000, 'train_loss': 0.372230169281755, 'last_train_loss': 0.19179978966712952}, {'num_samples': 60000, 'epoch': 3, 'batch_count': 15000, 'train_loss': 0.32247564417196833, 'last_train_loss': 0.30726122856140137}, {'num_samples': 60000, 'epoch': 4, 'batch_count': 15000, 'train_loss': 0.2959285915141232, 'last_train_loss': 0.2786743640899658}, {'num_samples': 60000, 'epoch': 5, 'batch_count': 15000, 'train_loss': 0.27712880933261197, 'last_train_loss': 0.2697388529777527}]

Validation stats: {'num_samples': 10000, 'Accuracy': tensor(0.8788), 'val_loss': 0.34675604103680596}
```
