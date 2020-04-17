# image similarity
This is a sample example of image similarity calculation. Both semantic and visually similarity are
introduced. A real estate example was used to recommend similar houses based on the query image
provided by users.

## Environment
* Python 3.6/3.7
* Apache Spark 2.x (This version needs to be same with the version you use to build Analytics Zoo)

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.

## Run Jupyter after pip install
```bash
export SPARK_DRIVER_MEMORY=10g
jupyter notebook --notebook-dir=./ --ip=* --no-browser
```

## Run Jupyter with prebuilt package
* Run `export SPARK_HOME=the root directory of Spark`.
* Run `export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package`.
* Run the following bash command to start the jupyter notebook. Change parameter settings as you need, e.g. `MASTER = local[1]`.
```bash
${ANALYTICS_ZOO_HOME}/bin/jupyter-with-zoo.sh \
    --master local[1] \
    --driver-memory 10g
```
