#!/bin/bash

set -e

export http_proxy=http://child-prc.intel.com:913
export https_proxy=http://child-prc.intel.com:913
#export no_proxy="10.239.45.10:8081"

#export http_proxy=http://proxy-chain.intel.com:911
#export https_proxy=https://proxy-chain.intel.com:912
#export no_proxy="10.239.47.210"

export JAVA_HOME=/opt/work/jdk8
export CLASSPATH=.:${JAVA_HOME}/lib:${JAVA_HOME}/jre/lib:${JAVA_HOME}/lib/tools.jar:${JAVA_HOME}/lib/dt.jar
export PATH=${JAVA_HOME}/bin/:${JAVA_HOME}/jre/bin:${PATH}
#export PATH=/opt/work/apache-maven-3.6.3/bin:$PATH 
#mvn --version
#export RUNTIME_K8S_SPARK_IMAGE=10.239.47.32/intelanalytics/hyper-zoo:latest

#docker login 10.239.45.10 --username admin --password '1234qwer!@#$QWER'
#export RUNTIME_K8S_SPARK_IMAGE=10.239.45.10/arda/hyper-zoo:latest

#export ANALYTICS_ZOO_ROOT=$WORKSPACE
#export FTP_URI=ftp://zoo:1234qwer@10.239.47.210

#chmod a+x pyzoo/zoo/examples/run-examples-test-k8s.sh

#echo "### start K8s example tests"
#pyzoo/zoo/examples/run-examples-test-k8s.sh
export SPARK_VERSION=3.1.2
if [ $SPARK_VERSION = '3.1.2' ]
then
  export SPARK_HOME=/opt/work/spark-3.1.2
  export EXAMPLE_FILE=/opt/spark/examples/jars/spark-examples_2.12-3.1.2.jar
  export RUNTIME_K8S_SPARK_IMAGE=10.239.45.10/arda/intelanalytics/hyper-zoo-spark-3.1.2:0.14.0-SNAPSHOT
else
  export SPARK_HOME=/opt/work/spark-2.4.6
  export EXAMPLE_FILE=/opt/spark/examples/jars/spark-examples_2.11-2.4.6.jar
  export RUNTIME_K8S_SPARK_IMAGE=10.239.45.10/arda/intelanalytics/hyper-zoo-spark-2.4.6:latest
fi

# set env
export BIGDL_VERSION=0.14.0-SNAPSHOT
export BIGDL_HOME=/opt/bigdl-${BIGDL_VERSION}
export BIGDL_CLASSPATH=${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:${BIGDL_HOME}/jars/bigdl-friesian-serving-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar
export RUNTIME_SPARK_MASTER=k8s://https://127.0.0.1:8443
export RUNTIME_K8S_SERVICE_ACCOUNT=spark
export RUNTIME_PERSISTENT_VOLUME_CLAIM=nfsvolumeclaim
export RUNTIME_DRIVER_HOST=172.16.0.200
export RUNTIME_DRIVER_PORT=54321
export RUNTIME_EXECUTOR_INSTANCES=1
export RUNTIME_EXECUTOR_CORES=16
export RUNTIME_EXECUTOR_MEMORY=16g
export RUNTIME_TOTAL_EXECUTOR_CORES=64
export RUNTIME_DRIVER_CORES=4
export RUNTIME_DRIVER_MEMORY=50g

# start python virtual env
# source activate bigdl
source activate newenv


echo "################## start autoestimator_pytorch.py  client "
starttime=`date +'%Y-%m-%d %H:%M:%S'`
${SPARK_HOME}/bin/spark-submit \
  --master ${RUNTIME_SPARK_MASTER} \
  --deploy-mode client \
  --conf spark.driver.host=${RUNTIME_DRIVER_HOST} \
  --conf spark.driver.port=${RUNTIME_DRIVER_PORT} \
  --conf spark.kubernetes.authenticate.driver.serviceAccountName=${RUNTIME_K8S_SERVICE_ACCOUNT} \
  --name analytics-zoo-autoestimator \
  --conf spark.kubernetes.container.image=${RUNTIME_K8S_SPARK_IMAGE} \
  --conf spark.executor.instances=${RUNTIME_EXECUTOR_INSTANCES} \
  --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
  --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/tmp \
  --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
  --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/tmp \
  --conf spark.kubernetes.driver.label.az=true \
  --conf spark.kubernetes.executor.label.az=true \
  --conf spark.kubernetes.node.selector.spark=true \
  --executor-cores ${RUNTIME_EXECUTOR_CORES} \
  --executor-memory ${RUNTIME_EXECUTOR_MEMORY} \
  --total-executor-cores ${RUNTIME_TOTAL_EXECUTOR_CORES} \
  --driver-cores ${RUNTIME_DRIVER_CORES} \
  --driver-memory ${RUNTIME_DRIVER_MEMORY} \
  --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
  --py-files local://${BIGDL_HOME}/python/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-serving-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local:///opt/bigdl-0.14.0-SNAPSHOT/examples/orca/automl/autoestimator/autoestimator_pytorch.py \
  --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
  --conf spark.sql.catalogImplementation='in-memory' \
  --conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  --conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  local:///opt/bigdl-0.14.0-SNAPSHOT/examples/orca/automl/autoestimator/autoestimator_pytorch.py
  
endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "################## end autoestimator_pytorch.py  client "
echo "run time is： "$((end_seconds-start_seconds))"s"



echo "################## start autoestimator_pytorch.py  cluster "
starttime=`date +'%Y-%m-%d %H:%M:%S'`
${SPARK_HOME}/bin/spark-submit \
  --master ${RUNTIME_SPARK_MASTER} \
  --deploy-mode cluster \
  --conf spark.kubernetes.authenticate.driver.serviceAccountName=${RUNTIME_K8S_SERVICE_ACCOUNT} \
  --name analytics-zoo-autoestimator \
  --conf spark.kubernetes.container.image=${RUNTIME_K8S_SPARK_IMAGE} \
  --conf spark.executor.instances=${RUNTIME_EXECUTOR_INSTANCES} \
  --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
  --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/tmp \
  --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
  --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/tmp \
  --conf spark.kubernetes.driver.label.az=true \
  --conf spark.kubernetes.executor.label.az=true \
  --conf spark.kubernetes.node.selector.spark=true \
  --executor-cores ${RUNTIME_EXECUTOR_CORES} \
  --executor-memory ${RUNTIME_EXECUTOR_MEMORY} \
  --total-executor-cores ${RUNTIME_TOTAL_EXECUTOR_CORES} \
  --driver-cores ${RUNTIME_DRIVER_CORES} \
  --driver-memory ${RUNTIME_DRIVER_MEMORY} \
  --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
  --py-files local://${BIGDL_HOME}/python/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-serving-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local:///opt/bigdl-0.14.0-SNAPSHOT/examples/orca/automl/autoestimator/autoestimator_pytorch.py \
  --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
  --conf spark.sql.catalogImplementation='in-memory' \
  --conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  --conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  local:///opt/bigdl-0.14.0-SNAPSHOT/examples/orca/automl/autoestimator/autoestimator_pytorch.py
  
endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "################## end autoestimator_pytorch.py  cluster "
echo "run time is： "$((end_seconds-start_seconds))"s"
