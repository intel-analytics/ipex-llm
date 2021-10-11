# TFPark

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
export MASTER=local[4]
python keras/keras_dataset.py
```

Using numpy.ndarray as data input
```bash
export MASTER=local[4]
python keras/keras_ndarray.py
```

## Run the KerasModel example with prebuilt package

Using TFDataset as data input

```bash
export ANALYTICS_ZOO_HOME=... # the directory where you extract the downloaded Analytics Zoo zip package
export SPARK_HOME=... # the root directory of Spark

bash $ANALYTICS_ZOO_HOME/bin/spark-submit-python-with-zoo.sh --master local[4] keras/keras_dataset.py
```

Using numpy.ndarray as data input
```bash
export ANALYTICS_ZOO_HOME=... # the directory where you extract the downloaded Analytics Zoo zip package
export SPARK_HOME=... # the root directory of Spark

bash $ANALYTICS_ZOO_HOME/bin/spark-submit-python-with-zoo.sh --master local[4] keras/keras_ndarray.py
```

## Run the TFEstimator example after pip install

Using TFDataset as data input
```bash
export MASTER=local[4]
export SPARK_DRIVER_MEMORY=2g
python estimator/estimator_dataset.py
```

Using FeatureSet as data input

```bash
# the directory to the training data, the sub-directory of IMAGE_PATH should be
# different classes each containing the images of that class.
# e.g.
# IMAGE_PATH=file:///cat_dog
# NUM_CLASSES=2
# /cat_dog
#    /cats
#       cat.001.jpg
#    /dogs
#       dog.001.jpg
IMAGE_PATH=... # file://... for local files and hdfs:// for hdfs files
NUM_CLASSES=..

export MASTER=local[4]
export SPARK_DRIVER_MEMORY=10g
python estimator/estimator_inception.py --image-path $IMAGE_PATH --num-classes $NUM_CLASSES
```

## Run the TFEstimator example with prebuilt package

Using TFDataset as data input
```bash
export ANALYTICS_ZOO_HOME=... # the directory where you extract the downloaded Analytics Zoo zip package
export SPARK_HOME=... # the root directory of Spark

bash $ANALYTICS_ZOO_HOME/bin/spark-submit-python-with-zoo.sh --master local[4] --driver-memory 2g estimator/estimator_dataset.py
```

Using FeatureSet as data input

```bash
export ANALYTICS_ZOO_HOME=... # the directory where you extract the downloaded Analytics Zoo zip package
export SPARK_HOME=... # the root directory of Spark


# the directory to the training data, the sub-directory of IMAGE_PATH should be
# different classes each containing the images of that class.
# e.g.
# IMAGE_PATH=file:///cat_dog
# NUM_CLASSES=2
# /cat_dog
#    /cats
#       cat.001.jpg
#    /dogs
#       dog.001.jpg
IMAGE_PATH=... # file://... for local files and hdfs:// for hdfs files
NUM_CLASSES=..


bash $ANALYTICS_ZOO_HOME/bin/spark-submit-python-with-zoo.sh --master local[4] --driver-memory 10g estimator/estimator_inception.py --image-path $IMAGE_PATH --num-classes $NUM_CLASSES
```

## Run the Training Example using TFOptimizer after pip install

```bash
export SPARK_MASTER=local[4]
export SPARK_DRIVER_MEMORY=2g
python tf_optimzer/train_lenet.py
```

## Run the Training Example using TFOptimizer with prebuilt package

```bash
export ANALYTICS_ZOO_HOME=... # the directory where you extract the downloaded Analytics Zoo zip package
export SPARK_HOME=... # the root directory of Spark

bash $ANALYTICS_ZOO_HOME/bin/spark-submit-python-with-zoo.sh --master local[4] --driver-memory 2g tf_optimizer/train_lenet.py
```

## Run the Evaluation Example using TFPredictor after pip install

```bash
export SPARK_MASTER=local[4]
export SPARK_DRIVER_MEMEORY=2g
python tf_optimizer/evaluate_lenet.py
```

## Run the Evaluation Example using TFPredictor with prebuilt package

```bash
export ANALYTICS_ZOO_HOME=... # the directory where you extract the downloaded Analytics Zoo zip package
export SPARK_HOME=... # the root directory of Spark

bash $ANALYTICS_ZOO_HOME/bin/spark-submit-python-with-zoo.sh --master local[4] --driver-memory 2g tf_optimizer/evaluate_lenet.py
```

## Run the GAN example after pip install

Please first install tensorflow_gan to run this example. (pip install tensorflow_probability==0.7.0 tensorflow_datasets==2.0.0 tensorflow_gan==2.0.0)

### Train and evaluation
```bash
export MASTER=local[1]
python gan/gan_train_and_evaluate.py
```
The training program will generate a TensorFlow checkpoint at /tmp/gan_model and every 1000 steps will generate 50 hand-written
digits and save them in a single image in the current directory.

The following is the generated image after 20000 steps.

![gan](./gan/image_20100.png)


## Run the GAN with prebuilt package

Please first install tensorflow_gan to run this example. (pip install tensorflow_gan==2.0.0)

### Training
```bash
export ANALYTICS_ZOO_HOME=... # the directory where you extract the downloaded Analytics Zoo zip package
export SPARK_HOME=... # the root directory of Spark

bash $ANALYTICS_ZOO_HOME/bin/spark-submit-python-with-zoo.sh --master local[1] gan/gan_train_and_evaluate.py
```

The training program will generate a TensorFlow checkpoint at /tmp/gan_model and every 1000 steps will generate 50 hand-written
digits and save them in a single image in the current directory.

The following is the generated image after 20000 steps.

![gan](./gan/image_20100.png)