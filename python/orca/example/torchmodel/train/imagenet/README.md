# Distributed pytorch on ImageNet dataset

This is an example to show you how to use analytics-zoo to train a pytorch model on Spark. 

# Requirements
* Python 3.7
* torch 1.5.0 or above
* torchvision 0.6.0 or above
* Apache Spark 2.4.3(pyspark)
* jep 3.9.0
* cloudpickle 1.6.0

## Prepare environments
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the enviroments, especially if you want to run on a yarn cluster(yarn-client mode only). 
```
conda create -n zoo python=3.7 #zoo is conda enviroment name, you can set another name you like.
conda activate zoo
pip install analytics-zoo==0.9.0.dev0 # or above
pip install jep==3.9.0
pip install cloudpickle==1.6.0
conda install pytorch torchvision cpuonly -c pytorch #command for linux
conda install pytorch torchvision -c pytorch #command for macOS
```

## Prepare Dataset
You need to sign up and download [ILSVRC2012_img_train.tar](http://image-net.org/download-images) and [ILSVRC2012_img_val.tar](http://image-net.org/download-images) manually, and unzip to the folder named `imagenet/train` and `imagenet/test`, (you should put the data to all driver and executor nodes).
For example, you can set data to `/tmp/imagenet` if your structure is like this:
```
/tmp/imagenet$ tree
.
└── imagenet
    └── train
        ├──n01440764
        |   ├──n01440764_10026.JPEG
        |    ...
        ...
    └── val
        ├──n01440764
        |   ├──n01440764_0000239.JPEG
        |    ...
        ...
```

## Run example
You can run this example on local mode and yarn client mode.

- Run with Spark Local mode
You can easily use the following commands to run this example:
    ```bash
    conda activate zoo
    export PYTHONHOME=[conda install path]/envs/zoo # use command "conda env list" to find the path of PYTHONEHOME.
    export ZOO_NUM_MKLTHREADS=all
    python main.py /tmp/imagenet
    ```

- Run with Yarn Client mode, export env `HADOOP_CONF_DIR`:  
    ```bash
    conda activate zoo
    export HADOOP_CONF_DIR=[path to your hadoop conf directory who has yarn-site.xml]
    python main.py /tmp/imagenet
    ```
    
In above commands
* data: the path to imagenet dataset.
* --batch-size: mini-batch size.
* --lr: learning rate.
* --epochs: number of epochs to train.
* --seed: seed for initializing training.
* --cores: number of CPUs to use.
* --nodes: number of nodes to use.
