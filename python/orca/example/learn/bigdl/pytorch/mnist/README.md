# Orca PyTorch estimator on MNIST dataset

This is an example to show you how to use analytics-zoo orca PyTorch Estimator to train a pytorch model on Spark.

# Requirements
* Python 3.7
* torch 1.5.0 or above
* torchvision 0.6.0 or above
* Apache Spark 2.4.3(pyspark)
* jep 3.9.0

## Prepare environments
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the enviroments, especially if you want to run on a yarn cluster(yarn-client mode only).
```
conda create -n zoo python=3.7 #zoo is conda enviroment name, you can set another name you like.
conda activate zoo
pip install analytics-zoo==0.9.0.dev0 # or above
pip install jep==3.9.0
conda install pytorch torchvision cpuonly -c pytorch #command for linux
conda install pytorch torchvision -c pytorch #command for macOS
```

## Prepare Dataset
If your nodes can access internet, the data will be downloaded to your dist automatically. Or you need to download
[train-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz), [train-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz), [t10k-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz) and [t10k-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz) manumally to some folder named `MNIST/raw`, and set --dir to it's parent dir. (you should put the data to all driver and executor nodes)
For example, you can set --dir to `/tmp/data` if your structure is like this:
```
/tmp/data$ tree
.
└── MNIST
    └── raw
        ├── t10k-images-idx3-ubyte.gz
        ├── t10k-labels-idx1-ubyte.gz
        ├── train-images-idx3-ubyte.gz
        └── train-labels-idx1-ubyte.gz
```

## Run example
You can run this example on local mode and yarn client mode.

- Run with Spark Local mode:
```bash
conda activate zoo
export PYTHONHOME=[conda install path]/envs/zoo # use command "conda env list" to find the path of PYTHONEHOME.
export ZOO_NUM_MKLTHREADS=4
python lenet_mnist.py
```

- Run with Yarn Client mode:
```bash
conda activate zoo
export HADOOP_CONF_DIR=[path to your hadoop conf directory which has yarn-site.xml]
export ZOO_NUM_MKLTHREADS=all
python lenet_mnist.py
```

In above commands
* --dir: the path to mnist dataset
* --batch-size: The mini-batch size on each executor.
* --test-batch-size: The test's mini-batch size on each executor.
* --lr: learning rate.
* --epochs: number of epochs to train.
* --seed: random seed.
* --save-model: for saving the current model.