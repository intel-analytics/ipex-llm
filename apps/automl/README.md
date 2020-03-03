# AutoML for time series forecasting
In this notebook, we use automl to do time series forecasting on NYC taxi dataset.

## Environment
* Python 3.6
* Apache Spark 2.4.3
* Jupyter Notebook
* Matplotlib

## Install Analytics Zoo AutoML
Analytics Zoo AutoML is still in experimental stage. And you need to manual build whl and pip 
it in local. 

1. First, download Analytics Zoo automl source code from [GitHub](https://github.com/intel-analytics/analytics-zoo/tree/automl):

2. Then build whl package for pip install. You may also refer to the doc [here](https://analytics-zoo.github.io/master/#DeveloperGuide/python/#build-whl-package-for-pip-install).
    ```bash
    bash analytics-zoo/pyzoo/dev/build.sh linux default
    ```
3. Create conda environment.
    1. Install [Conda](https://docs.conda.io/projects/conda/en/latest/commands/install.html).
    2. Create a new conda environment (with name "zoo_automl" for example).
    ```bash
    conda create -n zoo_automl python=3.6
    source activate zoo_automl
    ```
4. Install the whl built before. The `whl` locates in `analytics-zoo/pyzoo/dist`
    ```bash
    pip install analytics-zoo/pyzoo/dist/analytics_zoo-VERSION-py2.py3-none-PLATFORM_x86_64.whl[automl]
    ```
    Note that the extra dependencies (including `ray`, `psutil`, `aiohttp`, `setproctitle`, `scikit-learn`,`featuretools`, `tensorflow`, `pandas`, `requests`) will be installed by specifying `[automl]`.


## Run Jupyter after pip install
* install jupyter by `conda install jupyter`
* Run `export ANALYTICS_ZOO_HOME="path to analytics-zoo"`.
* Run `$ANALYTICS_ZOO_HOME/dist/bin/data/NAB/nyc_taxi/get_nyc_taxi.sh` to download the dataset. (It can also be downloaded from its [github](https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv)).
