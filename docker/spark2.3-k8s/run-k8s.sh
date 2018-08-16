export SPARK_HOME=/home/wickedspoon/Documents/work/spark-2.3.0-bin-hadoop2.7
$SPARK_HOME/bin/spark-submit \
    --master k8s://https://192.168.99.105:8443 \
    --deploy-mode cluster \
    --name bigdl-lenet \
    --class com.intel.analytics.bigdl.models.lenet.Train \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
    --conf spark.executor.instances=4 \
    --conf spark.executor.cores=4 \
    --conf spark.cores.max=16 \
    --conf spark.kubernetes.container.image=docker.io/spoonfree/wtf:latest \
    local:///opt/bigdl-0.6.0/lib/bigdl-SPARK_2.3-0.6.0-jar-with-dependencies.jar \
    -f /tmp/ \
    -b 128 \
    -e 2 \
    --checkpoint /tmp

