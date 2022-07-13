# Wide & Deep Recommendation for large scale data on Spark

This example demonstrates how to run a large-scale data recommendation task using a wide & deep model.

## Environment
* Apache Hadoop 2.7 or above
* Python 3.7
* tensorflow 2.9.1
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
If java is not installed, you can follow [this document](https://bigdl.readthedocs.io/en/latest/doc/UserGuide/python.html#install).

## Run Jupyter
You can start jupyter notebook with the follow command:
```
jupyter notebook
```

## Run Demo
If you don't have [Twitter Recsys Challenge 2021 dataset](https://recsys-twitter.com/data/show-downloads#), please run [`dummy_data_generating.py`](./dummy_data_generating.py) first (you can run `python dummy_data_generating.py` to get more infomation) and modify the file path in following steps.

After having a dataset, please run [`feature_engineering.ipynb`](./feature_engineering.ipynb) to process dataset for feature extraction.

Now everything is ready, run [`model_training.ipynb`](./model_training.ipynb) to get the recommander model.