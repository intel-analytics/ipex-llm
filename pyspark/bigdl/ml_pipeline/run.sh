BigDL_HOME=/home/wickedspoon/Documents/work/BigDL
SPARK_HOME=/home/wickedspoon/Documents/work/spark-2.0.1-bin-hadoop2.6
PYTHON_API_PATH=/home/wickedspoon/Documents/work/BigDL/dist/lib/bigdl-0.3.0-SNAPSHOT-python-api.zip
BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-0.3.0-SNAPSHOT-jar-with-dependencies.jar

export PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --notebook-dir=./ --ip=* --no-browser"


${SPARK_HOME}/bin/pyspark \
  --master local[1] \
  --driver-memory 4g \
  --properties-file ${BigDL_HOME}/dist/conf/spark-bigdl.conf \
  --py-files ${PYTHON_API_PATH} \
  --jars ${BigDL_JAR_PATH}\
  --conf spark.driver.extraClassPath=${BigDL_JAR_PATH}  \
  --conf spark.executor.extraClassPath=${BigDL_JAR_PATH}

