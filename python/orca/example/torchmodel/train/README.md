# Distributed pytorch

There are some examples to show you how to use analytics-zoo to distribute pytorch on Spark.
1. [MNIST](./train/mnist)
2. [Imagenet](./train/imagenet)
3. [Resnet finetune with NNFrame](./train/resnet_finetune)

# Requirements
* Python 3.7
* torch 1.5.0 or above
* torchvision 0.6.0 or above
* Apache Spark 2.4.6(pyspark)
* jep 3.9.0
* cloudpickle 1.6.0
* Java 1.8

## Prepare environments
We highly recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the enviroments, especially if you want to run on a yarn cluster(yarn-client mode only). 
```
conda create -n zoo python=3.7 #zoo is conda enviroment name, you can set another name you like.
conda activate zoo
pip install analytics-zoo==0.9.0.dev0 # or above
pip install jep==3.9.0 cloudpickle==1.6.0
conda install pytorch torchvision cpuonly -c pytorch #command for linux
conda install pytorch torchvision -c pytorch #command for macOS
```
If java is not installed, use command `java` to check if java is installed, you can use one of following commnads:  
1. system's package management system(like apt): `sudo apt-get install openjdk-8-jdk`.  
2. conda: `conda install openjdk=8.0.152`.
3. Manual installation: [oracle jdk](https://www.oracle.com/java/technologies/javase/javase-jdk8-downloads.html).
