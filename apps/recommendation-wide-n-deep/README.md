# Recommendation
This notebook demonstrates how to build a neural network recommendation system (Wide and Deep) with explict feedback using Analytics Zoo and BigDL on Spark. 


## Environment
* Python 3.5/3.6
* JDK 8
* Apache Spark 2.x (This version needs to be same with the version you use to build Analytics Zoo)
* Jupyter Notebook 4.1


## Install or download Analytics Zoo  
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.  


## Run after pip install
You can easily use the following commands to run this example:

    export SPARK_DRIVER_MEMORY=22g
    
    jupyter notebook --notebook-dir=./ --ip=* --no-browser 

See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-after-pip-install) for more running guidance after pip install. 


## Run with prebuild package
Run the following command for Spark local mode (MASTER=local[*]) or cluster mode:

    export SPARK_HOME=the root directory of Spark
    export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package

```
    ${ANALYTICS_ZOO_HOME}/bin/jupyter-with-zoo.sh \
    --master ${MASTER} \
    --driver-cores 4  \
    --driver-memory 22g  \
    --total-executor-cores 4  \
    --executor-cores 4  \
    --executor-memory 22g
```

See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-without-pip-install) for more running guidance without pip install. 
