SPARK_HOME=/home/ding/Downloads/spark-2.4.3-bin-hadoop2.7
MASTER=local[2]
PYTHON_API_ZIP_PATH=/home/ding/proj/clone-ding-zoo/analytics-zoo/scala/dllib/target/bigdl-dllib-2.0.0-SNAPSHOT-python-api.zip
BigDL_JAR_PATH=/home/ding/proj/clone-ding-zoo/analytics-zoo/scala/dllib/bigdl-dllib-2.0.0-SNAPSHOT-jar-with-dependencies.jar
PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH
${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --driver-cores 2  \
    --driver-memory 2g  \
    --total-executor-cores 2  \
    --executor-cores 2  \
    --executor-memory 4g \
    --py-files ${PYTHON_API_ZIP_PATH},/home/ding/proj/clone-ding-zoo/analytics-zoo/python/dllib/examples/lenet/lenet.py  \
    --properties-file /home/ding/proj/clone-ding-zoo/analytics-zoo/scala/dllib/src/main/resources/spark-bigdl.conf \
    --jars ${BigDL_JAR_PATH} \
    --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
    --conf spark.executor.extraClassPath=/home/ding/proj/clone-ding-zoo/analytics-zoo/scala/dllib/bigdl-dllib-2.0.0-SNAPSHOT-jar-with-dependencies.jar \
    /home/ding/proj/clone-ding-zoo/analytics-zoo/python/dllib/examples/lenet/lenet.py
