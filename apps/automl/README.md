# AutoML for time series forecasting
In this notebook, we use automl to do time series forecasting on NYC taxi dataset.

## Prepare environment

We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the enviroments.
```
conda create -n zoo python=3.6 #zoo is conda enviroment name, you can set another name you like.
source activate zoo
pip install analytics-zoo[automl]
```
Note that the extra dependencies (including `ray`, `psutil`, `aiohttp`, `setproctitle`, `scikit-learn`,`featuretools`, `tensorflow`, `pandas`, `requests`, `bayesian-optimization`) will be installed by specifying `[automl]`.  
You can also find the details of zoo installation [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) 

## Run Jupyter
* Install jupyter by `conda install jupyter`
* Run `export ANALYTICS_ZOO_HOME="path to analytics-zoo"`.
* Run `$ANALYTICS_ZOO_HOME/dist/bin/data/NAB/nyc_taxi/get_nyc_taxi.sh` to download the dataset. (It can also be downloaded from its [github](https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv)).
