# Distributed Face Generation on Spark

This example is migrated from [PROGRESSIVE GROWING OF GANS](https://pytorch.org/hub/facebookresearch_pytorch-gan-zoo_pgan/), and demonstrates how to run distributed inference using a pre-trained Pytorch Model.

# Requirements
* Python 3.7
* torch 1.5.0 or above
* torchvision 0.6.0 or above
* Apache Spark 2.4.6(pyspark)
* jep 3.9.1
* cloudpickle 1.6.0
* Java 1.8

## Prepare environments
We highly recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the enviroments, especially if you want to run on a yarn cluster(yarn-client mode only). 
```
conda create -n bigdl python=3.7 #bigdl is conda enviroment name, you can set another name you like.
conda activate bigdl
pip install orca dllib
pip install jep==3.9.1 cloudpickle==1.6.0
conda install pytorch torchvision cpuonly -c pytorch #command for linux
conda install pytorch torchvision -c pytorch #command for macOS
```
If java is not installed, use command `java` to check if java is installed, you can use one of following commnads:  
1. system's package management system(like apt): `sudo apt-get install openjdk-8-jdk`.  
2. conda: `conda install openjdk=8.0.152`.
3. Manual installation: [oracle jdk](https://www.oracle.com/java/technologies/javase/javase-jdk8-downloads.html).

## Run Jupyter
If you want to run spark local, just start jupyter notebook:
```
jupyter notebook
```

