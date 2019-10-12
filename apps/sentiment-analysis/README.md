# Sentiment Analysis
In this example, you will learn how to use Analytics Zoo to develop deep learning models for sentiment analysis including:
* How to load and review the IMDB dataset
* How to do word embedding with Glove
* How to build a CNN model for NLP
* How to build a LSTM model for NLP
* How to build a GRU model for NLP
* How to build a Bi-LSTM model for NLP
* How to build a CNN-LSTM model for NLP
* How to train deep learning models

## Environment
* Python 2.7/3.5/3.6
* Apache Spark 1.6.0/2.1.0 (This version needs to be same with the version you use to build Analytics Zoo)
* Numpy >= 1.16.0

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.

## Run after pip install
Start jupyter notebook as you normally do, e.g.
```
jupyter notebook --notebook-dir=./ --ip=* --no-browser```bash
```
See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-after-pip-install) for more running guidance after pip install.

## Run with prebuilt package
Run the following bash command to start the jupyter notebook. Change parameter settings as you need, ie `MASTER = local[physcial_core_number]`.
```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package

MASTER=local[*]
${ANALYTICS_ZOO_HOME}/bin/jupyter-with-zoo.sh \
    --master ${MASTER} \
    --driver-cores 4  \
    --driver-memory 12g  \
    --total-executor-cores 4  \
    --executor-cores 4  \
    --executor-memory 12g
```
See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-without-pip-install) for more running guidance without pip install.
