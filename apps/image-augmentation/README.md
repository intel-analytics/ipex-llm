# Image Augmentation
This is a simple example of image augmentation using Analytics ZOO API. We use various ways to transform images to augment the dataset.

## Environment
* Python 3.5/3.6 (numpy 1.11.1)
* Apache Spark 2.x (This version needs to be same with the version you use to build Analytics Zoo)

## Install or download Analytics Zoo
* Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.  

## Install OpenCV
* The example uses OpenCV library to save image. Please install it before run this example.
* You can use the following command:
```
apt-get install python-opencv
```

## Run after pip install
* You can easily use the following commands to run this example:
```
export SPARK_DRIVER_MEMORY=1g
jupyter notebook --notebook-dir=./ --ip=* --no-browser 
```

* See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-after-pip-install) for more running guidance after pip install. 


## Run with prebuild package
* Run the following command for Spark local mode (MASTER=local[*]) or cluster mode:
```bash
MASTER=local[*]
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
${ANALYTICS_ZOO_HOME}/bin/jupyter-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 1g  \
    --executor-memory 1g
```
* See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-without-pip-install) for more running guidance without pip install.
