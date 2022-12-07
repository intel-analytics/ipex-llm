# Wide & Deep Recommendation for large scale data on BigDL

This example demonstrates how to run a large-scale data recommendation task using a wide & deep model.

## Prepare environments
We highly recommend you use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environment, especially if you want to run on a yarn cluster. 
```
conda create -n bigdl python=3.7 #bigdl is conda environment name, you can set another name you like.
conda activate bigdl
pip install bigdl-orca[ray]
pip install bigdl-friesian
pip install tensorflow==2.9.1
```
Refer to [this document](https://bigdl.readthedocs.io/en/latest/doc/UserGuide/python.html#install) for more installation guides.

## Run Jupyter
You can start Jupyter notebook with the following command:
```
jupyter notebook
```

## Run Demo
This demo uses Twitter Recsys Challenge 2021 dataset, you can download it from [here](http://www.recsyschallenge.com/2021/). If you fail to download it, please run [`generate_dummy_data.py`](./generate_dummy_data.py) to generate a dummy dataset.

After having a dataset, modify the paths in following notebooks, both HDFS path and local path are acceptable.

Please run [`feature_engineering.ipynb`](./feature_engineering.ipynb) to process the dataset for feature extraction.

Now everything is ready, run [`model_training.ipynb`](./model_training.ipynb) to get the recommender model.