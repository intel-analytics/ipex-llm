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
export BIGDL_CONF=${BIGDL_HOME}/conf/spark-bigdl.conf
export BIGDL_PY_ZIP=`find ${BIGDL_HOME}/python -name bigdl-spark_*-python-api.zip`

# Check files
if [ ! -f ${BIGDL_CONF} ]; then
    "Cannot find BigDL configuration file, please check your BigDL download"
    exit 1
fi

if [ ! -f ${BIGDL_PY_ZIP} ]; then
    "Cannot find BigDL python zip file, please check your BigDL download"
    exit 1
fi

if [ ! -d ${BIGDL_HOME}/jars ]; then
    echo "Cannot find BigDL jar files, please check your BigDL download"
    exit 1
fi


${SPARK_HOME}/bin/spark-submit \
  --properties-file ${BIGDL_CONF} \
  --py-files ${BIGDL_PY_ZIP} \
  --conf spark.driver.extraClassPath=${BIGDL_HOME}/jars/* \
  --conf spark.executor.extraClassPath=${BIGDL_HOME}/jars/* \
  $*
