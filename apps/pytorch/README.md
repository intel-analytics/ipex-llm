# Distributed Face Generation on Spark

This example is migrated from [PROGRESSIVE GROWING OF GANS](https://pytorch.org/hub/facebookresearch_pytorch-gan-zoo_pgan/), and demonstrates how to run distributed inference using a pre-trained Pytorch Model.

## Requirements
* Python 3.6
* JDK 1.8
* Pytorch & TorchVision 1.1.0
* Apache Spark 2.4.3(pyspark)
* Analytics-Zoo 0.6.0-SNAPSHOT.dev6 and above
* Jupyter Notebook, matplotlib

## Prepare environments
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the enviroments, especially if you want to run on a yarn cluster(yarn-client mode only). 
```
conda create -n zoo python=3.6 #zoo is conda enviroment name, you can set another name you like.
conda activate zoo
pip install analytics-zoo==0.6.0.dev6 jupyter matplotlib
conda install pytorch-cpu torchvision-cpu -c pytorch
```

If you want install the latest analytics-zoo, you need to build the whl package by yourself, see [build-whl-package-for-pip-install](https://analytics-zoo.github.io/master/#DeveloperGuide/python/#build-whl-package-for-pip-install) for details.

## Run Jupyter
If you want to run spark local, just start jupyter notebook:
```
jupyter notebook
```

If you want to run on a yarn cluster(yarn-client mode only), export env `HADOOP_CONF_DIR` and `ZOO_CONDA_NAME` before starting jupyter notebook.
```
export HADOOP_CONF_DIR=[path to your hadoop conf directory]
export ZOO_CONDA_NAME=[conda environment name you just prepared above]
```

