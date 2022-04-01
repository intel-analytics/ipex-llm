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
export BIGDL_JARS="${BIGDL_HOME}/jars/*" | sed 's/ /,/g'
export BIGDL_CONF=${BIGDL_HOME}/conf/spark-bigdl.conf
export BIGDL_PY_ZIP=`find ${BIGDL_HOME}/python -name bigdl-spark_*-python-api.zip`

# Check files
if [ ! -f ${BIGDL_CONF} ]; then
    echo "Cannot find ${BIGDL_CONF}"
    exit 1
fi

#if [ ! -f $BIGDL_JAR ]; then
#    echo "Cannot find $BIGDL_JAR"
#    exit 1
#fi


${SPARK_HOME}/bin/spark-submit \
  --properties-file ${BIGDL_CONF} \
  --py-files ${BIGDL_PY_ZIP} \
  --jars ${BIGDL_JARS} \
  --conf spark.driver.extraClassPath=${BIGDL_HOME}/jars/* \
  --conf spark.executor.extraClassPath=${BIGDL_HOME}/jars/* \
  $*
