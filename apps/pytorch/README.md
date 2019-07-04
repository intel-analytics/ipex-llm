# Distributed Face Generation on Spark

This example is migrated from [PROGRESSIVE GROWING OF GANS](https://pytorch.org/hub/facebookresearch_pytorch-gan-zoo_pgan/), and demonstrates how to run distributed inference using a pre-trained Pytorch Model.

## Environment
* Python 3.6
* JDK 1.8
* Pytorch & TorchVision 1.1.0
* Apache Spark 2.4.3(pyspark)
* Analytics-Zoo 0.6.0-SNAPSHOT.dev4 and above
* Jupyter Notebook
* Ray, psutil(Will be removed in the future)

## Install Analytics Zoo
We recommend you to use Anaconda env to prepare the enviroments. Especially, you want to run on a yarn cluster(yarn-client mode). 
```
pip install analytics-zoo==0.6.0.dev4
```

## Run Jupyter
If you want to run spark local, just start jupyter notebook:
```
jupyter notebook
```

If you want to run on a yarn cluster(yarn-client mode), export env `HADOOP_CONF_DIR` and `ZOO_CONDA_NAME` before starting jupyter notebook.
```
export HADOOP_CONF_DIR=[path to your hadoop conf directory]
export ZOO_CONDA_NAME=[conda name intalled zoo]
```

