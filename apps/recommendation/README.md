# Demo Setup Guide
These two notebooks demostrate how to build neural network recommendation system with explict feedback using Analytics Zoo and BigDL on Spark. 

## Environment
* Python 2.7
* JDK 8
* Scala 2.11 
* Apache Spark 2.x
* Jupyter Notebook 4.1
* Zoo 0.1.0

## Steps to run the notebook
* Run `export ANALYTICS_ZOO_HOME=the home directory of the Analytics Zoo project`
* Run `export SPARK_HOME=the root directory of Spark`
* Run the following bash command to start the jupyter notebook. Change parameter settings as you need, ie MASTER = local\[physcial_core_number\]
```bash
MASTER=local[*]
bash ${ANALYTICS_ZOO_HOME}/scripts/jupyter-with-zoo.sh \
    --master ${MASTER} \
    --driver-cores 4  \
    --driver-memory 22g  \
    --total-executor-cores 4  \
    --executor-cores 4  \
    --executor-memory 22g \
```

