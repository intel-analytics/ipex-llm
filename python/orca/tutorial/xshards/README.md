# Summary
This directory contains examples to demonstrate how to use BigDL Orca [XShards](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/data-parallel-processing.html#xshards-distributed-data-parallel-python-processing) for distributed data preprocesing.

## Prepare the environment
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environment:

```
conda create -n bigdl python=3.7  # "bigdl" is the conda environment name, you can use any name you like.
conda activate bigdl
pip install --pre --upgrade bigdl-orca[ray]
```

## Prepare the data
- For auto_mpg.py, the dataset can be downloaded from [here](http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data).
- For diabetes.py, the dataset can be downloaded from [here](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv).
- For ionosphere.py, the dataset can be downloaded from [here](https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv).
- For tabular_playground_series.py, the dataset can be downloaded from [here](https://www.kaggle.com/code/remekkinas/tps-5-pytorch-nn-for-tabular-step-by-step/data?select=train.csv).
- For titanic.py, the dataset can be downloaded from [here](https://www.kaggle.com/code/chuanguy/titanic-data-processing-with-python-0-813/data?select=train.csv).
