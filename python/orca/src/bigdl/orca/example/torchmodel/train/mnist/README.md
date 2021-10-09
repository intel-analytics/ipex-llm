# Distributed pytorch on MNIST dataset

This is an example to show you how to use analytics-zoo to train a pytorch model on Spark. 

# Requirements
Before you run this example, the [requirements](../README.md#Requirements) should be installed.

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

- Run with Spark Local mode
You can easily use the following commands to run this example:
    ```bash
    conda activate zoo
    export PYTHONHOME=[conda install path]/envs/zoo # use command "conda env list" to find the path of PYTHONEHOME.
    export ZOO_NUM_MKLTHREADS=4
    python main.py
    ```

- Run with Yarn Client mode, upload data to hdfs first, export env `HADOOP_CONF_DIR` and `ZOO_CONDA_NAME`:  
    ```bash
    conda activate zoo
    export HADOOP_CONF_DIR=[path to your hadoop conf directory who has yarn-site.xml]
    export ZOO_CONDA_NAME=zoo #conda environment name you just prepared above
    export ZOO_NUM_MKLTHREADS=all
    python main.py
    ```
    
In above commands
* --dir: the path to mnist dataset
* --batch-size: The mini-batch size on each executor.
* --test-batch-size: The test's mini-batch size on each executor.
* --lr: learning rate.
* --epochs: number of epochs to train.
* --seed: random seed.
* --save-model: for saving the current model.
