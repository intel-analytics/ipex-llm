# Image Classification using TFNet

This example demonstrates how to do image classification inference using a pre-trained tensorflow checkpoint.

## Environment
* TensorFlow 1.10.0
* Apache Spark 2.x (This version needs to be same with the version you use to build Analytics Zoo)

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.

## Install Tensorflow-Slim image classification model library and down the checkpoint

Just clone the tensorflow models repository.

```shell
cd $HOME/workspace
git clone https://github.com/tensorflow/models/
```

Then download the InceptionV1 pre-trained checkpoint from [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)

## Run Jupyter after pip install

```bash
export SPARK_DRIVER_MEMORY=8g
jupyter notebook --notebook-dir=./ --ip=* --no-browser
```

## Run Jupyter with prebuilt package
* Run `export SPARK_HOME=the root directory of Spark`.
* Run `export ANALYTICS_ZOO_HOME=the dist directory under the Analytics Zoo project`.
* Prepare the training dataset from https://www.kaggle.com/c/dogs-vs-cats and extract it.
* Run the following bash command to start the jupyter notebook. Change parameter settings as you need, ie `MASTER = local[physcial_core_number]`.
```bash
MASTER=local[*]
${ANALYTICS_ZOO_HOME}/bin/jupyter-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 8g 
```