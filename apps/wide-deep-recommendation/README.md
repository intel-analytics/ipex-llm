# Wide & Deep Recommendation for large scale data on Spark

This example demonstrates how to run a large-scale data recommendation task using a wide & deep model.

## Environment
* Apache Hadoop 2.7 or above
* Python 3.7
* tensorflow 2.0.0 or above
* Apache Spark 2.4.6(pyspark)
* ray 1.9.2


## Prepare environments
We highly recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the enviroments, especially if you want to run on a yarn cluster(yarn-client mode only). 
```
conda create -n friesian python=3.7 #friesian is conda enviroment name, you can set another name you like.
conda activate friesian
pip install "bigdl-orca[ray]"
pip install bigdl-friesian
pip install tensorflow
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

## Run Demo
If you don't have [Twitter Recsys Challenge 2021 dataset](https://recsys-twitter.com/data/show-downloads#), please run [`dummy_data_generating.py`](./dummy_data_generating.py) first (you can run `python dummy_data_generating.py` to get more infomation) and modify the file path in following steps.

After having a dataset, please run [`feature_engineering.ipynb`](./feature_engineering.ipynb) to process dataset for feature extraction.

Now everything is ready, run [`model_training.ipynb`](./model_training.ipynb) to get the recommander model.