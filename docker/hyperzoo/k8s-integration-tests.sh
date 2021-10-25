#!/bin/bash
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


echo "################## start nnframes/imageinference.py  cluster "
starttime=`date +'%Y-%m-%d %H:%M:%S'`
${SPARK_HOME}/bin/spark-submit \
  --master ${RUNTIME_SPARK_MASTER} \
  --deploy-mode cluster \
  --conf spark.kubernetes.authenticate.driver.serviceAccountName=${RUNTIME_K8S_SERVICE_ACCOUNT} \
  --name analytics-zoo-nnframes-image-inference \
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
  --py-files local://${BIGDL_HOME}/python/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-serving-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local:///opt/bigdl-0.14.0-SNAPSHOT/examples/dllib/nnframes/imageInference/ImageInferenceExample.py \
  --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
  --conf spark.sql.catalogImplementation='in-memory' \
  --conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  --conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  local:///opt/bigdl-0.14.0-SNAPSHOT/examples/dllib/nnframes/imageInference/ImageInferenceExample.py \
  -m /tmp/data2/analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model \
  -f /tmp/data2/dogscats \
  --b 64
endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "################## end nnframes/imageinference.py  cluster "
echo "run time is： "$((end_seconds-start_seconds))"s"


echo "################## start nnframes/imageinference.py  client "
starttime=`date +'%Y-%m-%d %H:%M:%S'`
## client
${SPARK_HOME}/bin/spark-submit \
  --master ${RUNTIME_SPARK_MASTER} \
  --deploy-mode client \
  --conf spark.driver.host=${RUNTIME_DRIVER_HOST} \
  --conf spark.driver.port=${RUNTIME_DRIVER_PORT} \
  --conf spark.kubernetes.authenticate.driver.serviceAccountName=${RUNTIME_K8S_SERVICE_ACCOUNT} \
  --name analytics-zoo-nnframes-image-inference \
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
  --py-files local://${BIGDL_HOME}/python/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-serving-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local:///opt/bigdl-0.14.0-SNAPSHOT/examples/dllib/nnframes/imageInference/ImageInferenceExample.py \
  --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
  --conf spark.sql.catalogImplementation='in-memory' \
  --conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  --conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  local:///opt/bigdl-0.14.0-SNAPSHOT/examples/dllib/nnframes/imageInference/ImageInferenceExample.py \
  -m /tmp/data2/analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model \
  -f /tmp/data2/dogscats \
  --b 64
endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "################## end nnframes/imageinference.py  client "
echo "run time is： "$((end_seconds-start_seconds))"s"


echo "################## start nnframes/imagetransfer.py  cluster "
starttime=`date +'%Y-%m-%d %H:%M:%S'`
# b.nnframes image transfer learning
## cluster
${SPARK_HOME}/bin/spark-submit \
  --master ${RUNTIME_SPARK_MASTER} \
  --deploy-mode cluster \
  --conf spark.kubernetes.authenticate.driver.serviceAccountName=${RUNTIME_K8S_SERVICE_ACCOUNT} \
  --name analytics-zoo-nnframes-image-transfer-learning \
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
  --py-files local://${BIGDL_HOME}/python/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-serving-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local:///opt/bigdl-0.14.0-SNAPSHOT/examples/dllib/nnframes/imageTransferLearning/ImageTransferLearningExample.py \
  --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
  --conf spark.sql.catalogImplementation='in-memory' \
  --conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  --conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  local:///opt/bigdl-0.14.0-SNAPSHOT/examples/dllib/nnframes/imageTransferLearning/ImageTransferLearningExample.py \
  -m /tmp/data2/analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model \
  -f /tmp/data2/nnframes/samples \
  --b 64
endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "################## end nnframes/imagetransfer.py  cluster "
echo "run time is： "$((end_seconds-start_seconds))"s"


echo "################## start nnframes/imagetransfer.py  client "
starttime=`date +'%Y-%m-%d %H:%M:%S'`
## client
${SPARK_HOME}/bin/spark-submit \
  --master ${RUNTIME_SPARK_MASTER} \
  --deploy-mode client \
  --conf spark.driver.host=${RUNTIME_DRIVER_HOST} \
  --conf spark.driver.port=${RUNTIME_DRIVER_PORT} \
  --conf spark.kubernetes.authenticate.driver.serviceAccountName=${RUNTIME_K8S_SERVICE_ACCOUNT} \
  --name analytics-zoo-nnframes-image-transfer-learning \
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
  --py-files local://${BIGDL_HOME}/python/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-serving-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local:///opt/bigdl-0.14.0-SNAPSHOT/examples/dllib/nnframes/imageTransferLearning/ImageTransferLearningExample.py \
  --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
  --conf spark.sql.catalogImplementation='in-memory' \
  --conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  --conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  local:///opt/bigdl-0.14.0-SNAPSHOT/examples/dllib/nnframes/imageTransferLearning/ImageTransferLearningExample.py \
  -m /tmp/data2/analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model \
  -f /tmp/data2/nnframes/samples \
  --b 64
endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "################## end nnframes/imagetransfer.py  client "
echo "run time is： "$((end_seconds-start_seconds))"s"

  

echo "################## start spark_pandas.py  client "
starttime=`date +'%Y-%m-%d %H:%M:%S'`
#  spark_pandas.py
## client
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
  --py-files local://${BIGDL_HOME}/python/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-serving-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local:///opt/bigdl-0.14.0-SNAPSHOT/examples/orca/data/spark_pandas.py \
  --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
  --conf spark.sql.catalogImplementation='in-memory' \
  --conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  --conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  local:///opt/bigdl-0.14.0-SNAPSHOT/examples/orca/data/spark_pandas.py \
  -f /bigdl2.0/data/nyc_taxi.csv
endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "################## end spark_pandas.py  client "
echo "run time is： "$((end_seconds-start_seconds))"s"

echo "################## start spark_pandas.py  cluster "
starttime=`date +'%Y-%m-%d %H:%M:%S'`
##cluster
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
  --py-files local://${BIGDL_HOME}/python/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-serving-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local:///opt/bigdl-0.14.0-SNAPSHOT/examples/orca/data/spark_pandas.py \
  --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
  --conf spark.sql.catalogImplementation='in-memory' \
  --conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  --conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  local:///opt/bigdl-0.14.0-SNAPSHOT/examples/orca/data/spark_pandas.py \
  -f /bigdl2.0/data/nyc_taxi.csv

endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "################## end spark_pandas.py  cluster "
echo "run time is： "$((end_seconds-start_seconds))"s"


# 13 super_resolution.py
## super_resolution.py client
echo "################## start super_resolution.py  client "
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
  --conf spark.kubernetes.driverEnv.http_proxy=${http_proxy} \
  --conf spark.kubernetes.driverEnv.https_proxy=${https_proxy} \
  --conf spark.kubernetes.executorEnv.http_proxy=${http_proxy} \
  --conf spark.kubernetes.executorEnv.https_proxy=${https_proxy} \
  --executor-cores ${RUNTIME_EXECUTOR_CORES} \
  --executor-memory ${RUNTIME_EXECUTOR_MEMORY} \
  --total-executor-cores ${RUNTIME_TOTAL_EXECUTOR_CORES} \
  --driver-cores ${RUNTIME_DRIVER_CORES} \
  --driver-memory ${RUNTIME_DRIVER_MEMORY} \
  --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
  --py-files local://${BIGDL_HOME}/python/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-serving-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local:///opt/bigdl-0.14.0-SNAPSHOT/examples/orca/learn/pytorch/super_resolution/super_resolution.py \
  --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
  --conf spark.sql.catalogImplementation='in-memory' \
  --conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  --conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  local:///opt/bigdl-0.14.0-SNAPSHOT/examples/orca/learn/pytorch/super_resolution/super_resolution.py \
  --backend torch_distributed

endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "################## end super_resolution.py  client "
echo "run time is： "$((end_seconds-start_seconds))"s"


echo "################## start super_resolution.py  cluster "
starttime=`date +'%Y-%m-%d %H:%M:%S'`
## super_resolution.py cluster
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
  --conf spark.kubernetes.driverEnv.http_proxy=${http_proxy} \
  --conf spark.kubernetes.driverEnv.https_proxy=${https_proxy} \
  --conf spark.kubernetes.executorEnv.http_proxy=${http_proxy} \
  --conf spark.kubernetes.executorEnv.https_proxy=${https_proxy} \
  --executor-cores ${RUNTIME_EXECUTOR_CORES} \
  --executor-memory ${RUNTIME_EXECUTOR_MEMORY} \
  --total-executor-cores ${RUNTIME_TOTAL_EXECUTOR_CORES} \
  --driver-cores ${RUNTIME_DRIVER_CORES} \
  --driver-memory ${RUNTIME_DRIVER_MEMORY} \
  --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
  --py-files local://${BIGDL_HOME}/python/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-serving-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local:///opt/bigdl-0.14.0-SNAPSHOT/examples/orca/learn/pytorch/super_resolution/super_resolution.py \
  --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
  --conf spark.sql.catalogImplementation='in-memory' \
  --conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  --conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  local:///opt/bigdl-0.14.0-SNAPSHOT/examples/orca/learn/pytorch/super_resolution/super_resolution.py  \
  --backend torch_distributed
endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "################## end super_resolution.py  cluster "
echo "run time is： "$((end_seconds-start_seconds))"s"


# cifar10.py
echo "################## start cifar10  client "
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
  --py-files local://${BIGDL_HOME}/python/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-serving-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local:///opt/bigdl-0.14.0-SNAPSHOT/examples/orca/learn/pytorch/cifar10/cifar10.py \
  --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
  --conf spark.sql.catalogImplementation='in-memory' \
  --conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  --conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  local:///opt/bigdl-0.14.0-SNAPSHOT/examples/orca/learn/pytorch/cifar10/cifar10.py \
  --backend torch_distributed
endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "################## end cifar10.py client "
echo "run time is： "$((end_seconds-start_seconds))"s"
  
echo "################## start cifar10.py  cluster "
starttime=`date +'%Y-%m-%d %H:%M:%S'`
##  cifar10.py cluster
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
  --conf spark.kubernetes.driverEnv.http_proxy=${http_proxy} \
  --conf spark.kubernetes.driverEnv.https_proxy=${https_proxy} \
  --conf spark.kubernetes.executorEnv.http_proxy=${http_proxy} \
  --conf spark.kubernetes.executorEnv.https_proxy=${https_proxy} \
  --executor-cores ${RUNTIME_EXECUTOR_CORES} \
  --executor-memory ${RUNTIME_EXECUTOR_MEMORY} \
  --total-executor-cores ${RUNTIME_TOTAL_EXECUTOR_CORES} \
  --driver-cores ${RUNTIME_DRIVER_CORES} \
  --driver-memory ${RUNTIME_DRIVER_MEMORY} \
  --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
  --py-files local://${BIGDL_HOME}/python/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-serving-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local:///opt/bigdl-0.14.0-SNAPSHOT/examples/orca/learn/pytorch/cifar10/cifar10.py \
  --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
  --conf spark.sql.catalogImplementation='in-memory' \
  --conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  --conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  local:///opt/bigdl-0.14.0-SNAPSHOT/examples/orca/learn/pytorch/cifar10/cifar10.py \
  --backend torch_distributed
endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "################## end cifar10.py  cluster "
echo "run time is： "$((end_seconds-start_seconds))"s"



echo "################## start transfer_learning.py  client "
starttime=`date +'%Y-%m-%d %H:%M:%S'`
## transfer_learning.py
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
  --conf spark.kubernetes.driverEnv.http_proxy=${http_proxy} \
  --conf spark.kubernetes.driverEnv.https_proxy=${https_proxy} \
  --conf spark.kubernetes.executorEnv.http_proxy=${http_proxy} \
  --conf spark.kubernetes.executorEnv.https_proxy=${https_proxy} \
  --executor-cores ${RUNTIME_EXECUTOR_CORES} \
  --executor-memory ${RUNTIME_EXECUTOR_MEMORY} \
  --total-executor-cores ${RUNTIME_TOTAL_EXECUTOR_CORES} \
  --driver-cores ${RUNTIME_DRIVER_CORES} \
  --driver-memory ${RUNTIME_DRIVER_MEMORY} \
  --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
  --py-files local://${BIGDL_HOME}/python/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-serving-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local:///opt/bigdl-0.14.0-SNAPSHOT/examples/orca/learn/tf/transfer_learning/transfer_learning.py \
  --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
  --conf spark.sql.catalogImplementation='in-memory' \
  --conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  --conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  local:///opt/bigdl-0.14.0-SNAPSHOT/examples/orca/learn/tf/transfer_learning/transfer_learning.py

endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "################## end transfer_learning.py client "
echo "run time is： "$((end_seconds-start_seconds))"s"

echo "################## start transfer_learning.py  cluster "
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
  --conf spark.kubernetes.driverEnv.http_proxy=${http_proxy} \
  --conf spark.kubernetes.driverEnv.https_proxy=${https_proxy} \
  --conf spark.kubernetes.executorEnv.http_proxy=${http_proxy} \
  --conf spark.kubernetes.executorEnv.https_proxy=${https_proxy} \
  --executor-cores ${RUNTIME_EXECUTOR_CORES} \
  --executor-memory ${RUNTIME_EXECUTOR_MEMORY} \
  --total-executor-cores ${RUNTIME_TOTAL_EXECUTOR_CORES} \
  --driver-cores ${RUNTIME_DRIVER_CORES} \
  --driver-memory ${RUNTIME_DRIVER_MEMORY} \
  --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
  --py-files local://${BIGDL_HOME}/python/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-serving-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local:///opt/bigdl-0.14.0-SNAPSHOT/examples/orca/learn/tf/transfer_learning/transfer_learning.py \
  --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
  --conf spark.sql.catalogImplementation='in-memory' \
  --conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  --conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  local:///opt/bigdl-0.14.0-SNAPSHOT/examples/orca/learn/tf/transfer_learning/transfer_learning.py

endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "################## end transfer_learning.py  cluster "
echo "run time is： "$((end_seconds-start_seconds))"s"


echo "################## start basic_text_classification.py  client "
starttime=`date +'%Y-%m-%d %H:%M:%S'`
## basic_text_classification.py
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
  --py-files local://${BIGDL_HOME}/python/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-serving-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local:///opt/bigdl-0.14.0-SNAPSHOT/examples/orca/learn/tf/basic_text_classification/basic_text_classification.py \
  --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
  --conf spark.sql.catalogImplementation='in-memory' \
  --conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  --conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  local:///opt/bigdl-0.14.0-SNAPSHOT/examples/orca/learn/tf/basic_text_classification/basic_text_classification.py

endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "################## end basic_text_classification.py  client "
echo "run time is： "$((end_seconds-start_seconds))"s"


echo "################## start basic_text_classification.py  cluster "
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
  --conf spark.kubernetes.driverEnv.http_proxy=${http_proxy} \
  --conf spark.kubernetes.driverEnv.https_proxy=${https_proxy} \
  --conf spark.kubernetes.executorEnv.http_proxy=${http_proxy} \
  --conf spark.kubernetes.executorEnv.https_proxy=${https_proxy} \
  --executor-cores ${RUNTIME_EXECUTOR_CORES} \
  --executor-memory ${RUNTIME_EXECUTOR_MEMORY} \
  --total-executor-cores ${RUNTIME_TOTAL_EXECUTOR_CORES} \
  --driver-cores ${RUNTIME_DRIVER_CORES} \
  --driver-memory ${RUNTIME_DRIVER_MEMORY} \
  --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
  --py-files local://${BIGDL_HOME}/python/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-serving-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local:///opt/bigdl-0.14.0-SNAPSHOT/examples/orca/learn/tf/basic_text_classification/basic_text_classification.py \
  --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
  --conf spark.sql.catalogImplementation='in-memory' \
  --conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  --conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar:local://${BIGDL_HOME}/jars/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
  local:///opt/bigdl-0.14.0-SNAPSHOT/examples/orca/learn/tf/basic_text_classification/basic_text_classification.py

endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "################## end basic_text_classification.py  cluster "
echo "run time is： "$((end_seconds-start_seconds))"s"
