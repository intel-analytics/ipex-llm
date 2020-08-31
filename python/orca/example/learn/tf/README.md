# Orca TF Estimator

This is an example to demonstrate how to use Analytics-Zoo's Orca TF Estimator API to run distributed
Tensorflow and Keras on Spark.

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.

## Environment Preparation
```
pip install tensorflow==1.15 tensorflow-datasets==2.0
pip install psutil
```

## Model Preparation

In this example, we will use the **slim** library to construct the model. You can
clone it [here](https://github.com/tensorflow/models/tree/master/research/slim) and add
the `research/slim` directory to `PYTHONPATH`.

```bash
git clone https://github.com/tensorflow/models/
export PYTHONPATH=$PWD/models/research/slim:$PYTHONPATH
```

## Run tf graph model example after pip install

```bash
python lenet_mnist_graph.py
```
## Run tf graph model example with prebuilt package

```bash
export ANALYTICS_ZOO_HOME=... # the directory where you extract the downloaded Analytics Zoo zip package
export SPARK_HOME=... # the root directory of Spark
bash $ANALYTICS_ZOO_HOME/bin/spark-submit-python-with-zoo.sh --master local[4] lenet_mnist_graph.py
```

## Run tf keras model example after pip install
```bash
python lenet_mnist_keras.py
```

## Run tf keras model example with prebuilt package
```bash
export ANALYTICS_ZOO_HOME=... # the directory where you extract the downloaded Analytics Zoo zip package
export SPARK_HOME=... # the root directory of Spark
bash $ANALYTICS_ZOO_HOME/bin/spark-submit-python-with-zoo.sh --master local[4] lenet_mnist_keras.py
```