# Image Classification using TFNet

This example demonstrates how to do image classification inference using a pre-trained tensorflow checkpoint.

## Environment
* Tensorflow 1.8.0
* Apache Spark 1.6.0/2.1.0 (This version needs to be same with the version you use to build Analytics Zoo)

## Install Tensorflow-Slim image classification model library and down the checkpoint

Just clone the tensorflow models repository.

```shell
cd $HOME/workspace
git clone https://github.com/tensorflow/models/
```

Then download the InceptionV1 pre-trained checkpoint from [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)


## Run with Jupyter
* Download Analytics Zoo and build it.
* Run `export SPARK_HOME=the root directory of Spark`.
* Run `export ANALYTICS_ZOO_HOME=the dist directory under the Analytics Zoo project`.
* Prepare the training dataset from https://www.kaggle.com/c/dogs-vs-cats and extract it.
* Run the following bash command to start the jupyter notebook. Change parameter settings as you need, ie `MASTER = local[physcial_core_number]`.
```bash
MASTER=local[*]
${ANALYTICS_ZOO_HOME}/bin/jupyter-with-zoo.sh \
    --master ${MASTER} \
    --driver-cores 2  \
    --driver-memory 8g  \
    --total-executor-cores 2  \
    --executor-cores 2  \
    --executor-memory 8g
```