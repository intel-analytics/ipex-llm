#!/bin/bash

# Check environment variables
if [ -z "${BIGDL_HOME}" ]; then
    echo "Please set BIGDL_HOME environment variable"
    exit 1
fi

if [ -z "${SPARK_HOME}" ]; then
    echo "Please set SPARK_HOME environment variable"
    exit 1
fi

#setup paths
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --notebook-dir=./ --ip=* --no-browser --NotebookApp.token=''"
export BIGDL_JAR_NAME=`ls ${BIGDL_HOME}/lib/ | grep jar-with-dependencies.jar`
export BIGDL_JAR="${BIGDL_HOME}/lib/$BIGDL_JAR_NAME"
export BIGDL_PY_ZIP_NAME=`ls ${BIGDL_HOME}/lib/ | grep python-api.zip`
export BIGDL_PY_ZIP="${BIGDL_HOME}/lib/$BIGDL_PY_ZIP_NAME"
export BIGDL_CONF=${BIGDL_HOME}/conf/spark-bigdl.conf
export PYTHONPATH=${BIGDL_PY_ZIP}:${PYTHONPATH}

# Check files
if [ ! -f ${BIGDL_CONF} ]; then
    echo "Cannot find ${BIGDL_CONF}"
    exit 1
fi

if [ ! -f ${BIGDL_PY_ZIP} ]; then
    echo ${BIGDL_PY_ZIP}
    echo "Cannot find ${BIGDL_PY_ZIP}"
    exit 1
fi

if [ ! -f $BIGDL_JAR ]; then
    echo "Cannot find $BIGDL_JAR"
    exit 1
fi

${SPARK_HOME}/bin/pyspark \
  --properties-file ${BIGDL_CONF} \
  --py-files ${BIGDL_PY_ZIP} \
  --jars ${BIGDL_JAR} \
  --conf spark.driver.extraClassPath=${BIGDL_JAR} \
  --conf spark.executor.extraClassPath=${BIGDL_JAR} \
  --conf spark.sql.catalogImplementation='in-memory' \
  $*
