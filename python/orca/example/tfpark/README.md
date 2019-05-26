# TFPark API

This is an example to demonstrate how to use Analytics-Zoo's TFPark API to run distributed
Tensorflow and Keras on Spark/BigDL.

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.

## Model Preparation

In this example, we will use the **slim** library to construct the model. You can
clone it [here](https://github.com/tensorflow/models/tree/master/research/slim) and
the `research/slim` directory to `PYTHONPATH`.

```bash

git clone https://github.com/tensorflow/models/

export PYTHONPATH=$PWD/models/research/slim:$PYTHONPATH
```


## Run the KerasModel example after pip install

Using TFDataset as data input

```bash
python keras_dataset.py
```

Using numpy.ndarray as data input
```bash
python keras_ndarray.py
```

## Run the KerasModel example with prebuilt package

Using TFDataset as data input

```bash
export ANALYTICS_ZOO_HOME=... # the directory where you extract the downloaded Analytics Zoo zip package
export SPARK_HOME=... # the root directory of Spark

sh $ANALYTICS_ZOO_HOME/bin/spark-submit-with-zoo.sh --master local[4] keras_dataset.py
```

Using numpy.ndarray as data input
```bash
export ANALYTICS_ZOO_HOME=... # the directory where you extract the downloaded Analytics Zoo zip package
export SPARK_HOME=... # the root directory of Spark

sh $ANALYTICS_ZOO_HOME/bin/spark-submit-with-zoo.sh --master local[4] keras_ndarray.py
```

## Run the TFEstimator example after pip install

Using TFDataset as data input
```bash
python estimator_dataset.py
```

Using FeatureSet as data input

```bash
# the directory to the training data, the sub-directory of IMAGE_PATH should be
# different classes each containing the images of that class.
# e.g.
# IMAGE_PATH=/cat_dog
# NUM_CLASSES=2
# /cat_dog
#    /cats
#       cat.001.jpg
#    /dogs
#       dog.001.jpg
IMAGE_PATH=...
NUM_CLASSES=..


python estimator_inception.py --image-path $IMAGE_PATH --num-classes $NUM_CLASSES
```

## Run the TFEstimator example with prebuilt package

Using TFDataset as data input
```bash
export ANALYTICS_ZOO_HOME=... # the directory where you extract the downloaded Analytics Zoo zip package
export SPARK_HOME=... # the root directory of Spark

sh $ANALYTICS_ZOO_HOME/bin/spark-submit-with-zoo.sh --master local[4] estimator_dataset.py
```

Using FeatureSet as data input

```bash
export ANALYTICS_ZOO_HOME=... # the directory where you extract the downloaded Analytics Zoo zip package
export SPARK_HOME=... # the root directory of Spark


# the directory to the training data, the sub-directory of IMAGE_PATH should be
# different classes each containing the images of that class.
# e.g.
# IMAGE_PATH=/cat_dog
# NUM_CLASSES=2
# /cat_dog
#    /cats
#       cat.001.jpg
#    /dogs
#       dog.001.jpg
IMAGE_PATH=...
NUM_CLASSES=..


sh $ANALYTICS_ZOO_HOME/bin/spark-submit-with-zoo.sh --master local[4] estimator_inception.py --image-path $IMAGE_PATH --num-classes $NUM_CLASSES
```