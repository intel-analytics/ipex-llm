# Autograd Examples
There are two examples that illustrate how to define a custom loss function and ```Lambda``` layer.

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.

## Run after pip install
You can easily use the following commands to run this example:
```
export SPARK_DRIVER_MEMORY=2g
python path/to/custom.py
python path/to/customloss.py
```

See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-after-pip-install) for more running guidance after pip install.

## Run with prebuilt package
Run the following command for Spark local mode (MASTER=local[*]) or cluster mode:
```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
MASTER=...
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master ${MASTER}\
    --driver-memory 2g \
    --executor-memory 2g \
    path/to/custom.py
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master ${MASTER}\
    --driver-memory 2g \
    --executor-memory 2g \
    path/to/customloss.py
```
See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-without-pip-install) for more running guidance without pip install.

