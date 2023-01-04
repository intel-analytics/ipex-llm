# Summary
This directory contains examples to demonstrate how to use BigDL Orca [XShards](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/data-parallel-processing.html#xshards-distributed-data-parallel-python-processing) for distributed data preprocesing.

## Prepare the environment and start the notebook for auto_mpg.ipynb, diabetes.ipynb, ionosphere.ipynb, tabular_playground_series.ipynb, titanic.ipynb, house_price_analysis.ipynb
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environment:

```
export SPAKR_HOME=your_spark3_home
conda create -n bigdl python=3.7  # "bigdl" is the conda environment name, you can use any name you like.
conda activate bigdl
pip install --pre --upgrade bigdl-orca-spark3
jupyter notebook
```

## Prepare environment for answer_correctness_analysis.ipynb
```
export SPAKR_HOME=your_spark3_home
conda create -n bigdl python=3.7  # "bigdl" is the conda environment name, you can use any name you like.
conda activate bigdl
pip install --pre --upgrade bigdl-orca-spark3[ray]
jupyter notebook
```


## Prepare the data
- For auto_mpg.ipynb, the dataset can be downloaded from [here](http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data).
- For diabetes.ipynb, the dataset can be downloaded from [here](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv).
- For ionosphere.ipynb, the dataset can be downloaded from [here](https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv).
- For tabular_playground_series.ipynb, the dataset can be downloaded from [here](https://www.kaggle.com/code/remekkinas/tps-5-pytorch-nn-for-tabular-step-by-step/data?select=train.csv).
- For titanic.ipynb, the dataset can be downloaded from [here](https://www.kaggle.com/code/chuanguy/titanic-data-processing-with-python-0-813/data?select=train.csv).
- For house_price_analysis.ipynb, the dataset can be downloaded from [here](https://www.kaggle.com/code/pmarcelino/comprehensive-data-exploration-with-python/data?select=train.csv)
- For answer_correctness_analysis.ipynb, the dataset can be downloaded from [here](https://www.kaggle.com/competitions/riiid-test-answer-prediction/data?select=train.csv)
