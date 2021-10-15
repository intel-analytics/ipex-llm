#!/bin/bash
set -x
#apt-get update
#apt-get install -y openjdk-11-jdk
cd /ppml/docker-occlum

cp /ppml/docker-occlum/spark-2.4.6-bin-hadoop2.7/jars/spark-network-common_2.11-2.4.6.jar /ppml/docker-occlum/spark-network-common_2.11-2.4.6.jar
BLUE='\033[1;34m'
NC='\033[0m'
occlum_glibc=/opt/occlum/glibc/lib/

init_instance() {
    # Init Occlum instance
    postfix=$1
    rm -rf occlum_instance_$postfix && mkdir occlum_instance_$postfix
    cd occlum_instance_$postfix
    occlum init
    new_json="$(jq '.resource_limits.user_space_size = "64000MB" |
        .resource_limits.max_num_of_threads = 512 |
        .process.default_heap_size = "128MB" |
        .resource_limits.kernel_space_heap_size="256MB" |
        .process.default_mmap_size = "50000MB" |
        .entry_points = [ "/usr/lib/jvm/java-11-openjdk-amd64/bin" ] |
        .env.default = [ "LD_LIBRARY_PATH=/usr/lib/jvm/java-11-openjdk-amd64/lib/server:/usr/lib/jvm/java-11-openjdk-amd64/lib:/usr/lib/jvm/java-11-openjdk-amd64/../lib:/lib","SPARK_CONF_DIR=/bin/conf","SPARK_ENV_LOADED=1","PYTHONHASHSEED=0","SPARK_HOME=/bin","SPARK_SCALA_VERSION=2.12","SPARK_JARS_DIR=/bin/jars","LAUNCH_CLASSPATH=/bin/jars/*",""]' Occlum.json)" && \
    echo "${new_json}" > Occlum.json
}

build_spark() {
    # Copy JVM and class file into Occlum instance and build
    mkdir -p image/usr/lib/jvm
    cp -r /usr/lib/jvm/java-11-openjdk-amd64 image/usr/lib/jvm
    cp /lib/x86_64-linux-gnu/libz.so.1 image/lib
    cp /lib/x86_64-linux-gnu/libz.so.1 image/$occlum_glibc
    cp $occlum_glibc/libdl.so.2 image/$occlum_glibc
    cp $occlum_glibc/librt.so.1 image/$occlum_glibc
    cp $occlum_glibc/libm.so.6 image/$occlum_glibc
    cp $occlum_glibc/libnss_files.so.2 image/$occlum_glibc
    cp -rf ../spark-2.4.6-bin-hadoop2.7/* image/bin/
    cp -rf ../hosts image/etc/
    cp -rf /etc/ssl image/etc/
    cp -rf /etc/passwd image/etc/
    cp -rf /etc/group image/etc/
    cp -rf /etc/java-11-openjdk image/etc/
    cp -rf ../bigdl-${BIGDL_VERSION}-jar-with-dependencies.jar image/bin/jars
    cp -rf ../cifar image/bin/
    /opt/occlum/start_aesm.sh
    occlum build
}

run_spark_test() {
    init_instance spark
    build_spark
    echo -e "${BLUE}occlum run spark${NC}"
    echo -e "${BLUE}logfile=$log${NC}"
    occlum run /usr/lib/jvm/java-11-openjdk-amd64/bin/java \
                -XX:-UseCompressedOops -XX:MaxMetaspaceSize=256m \
                -XX:ActiveProcessorCount=192 \
                -Divy.home="/tmp/.ivy" \
                -Dos.name="Linux" \
                -cp '/bin/conf/:/bin/jars/*' -Xmx10g org.apache.spark.deploy.SparkSubmit --jars /bin/examples/jars/spark-examples_2.11-2.4.6.jar,/bin/examples/jars/scopt_2.11-3.7.0.jar --class org.apache.spark.examples.SparkPi spark-internal
}

run_spark_bigdl(){
    init_instance spark
    build_spark
    echo -e "${BLUE}occlum run spark${NC}"
    echo -e "${BLUE}logfile=$log${NC}"
    occlum run /usr/lib/jvm/java-11-openjdk-amd64/bin/java \
                -XX:-UseCompressedOops -XX:MaxMetaspaceSize=256m \
                -XX:ActiveProcessorCount=24 \
                -Divy.home="/tmp/.ivy" \
                -Dos.name="Linux" \
                -cp '/bin/conf/:/bin/jars/*'  -Xmx10g org.apache.spark.deploy.SparkSubmit --jars /bin/examples/jars/spark-examples_2.11-2.4.6.jar,/bin/examples/jars/scopt_2.11-3.7.0.jar \
                --master 'local[4]' \
                --conf spark.driver.port=10027 \
                --conf spark.scheduler.maxRegisteredResourcesWaitingTime=5000000 \
                --conf spark.worker.timeout=600 \
                --conf spark.executor.extraClassPath=/bin/jars/bigdl-0.13.0-jar-with-dependencies.jar \
                --conf spark.driver.extraClassPath=/bin/jars/bigdl-0.13.0-jar-with-dependencies.jar \
                --conf spark.starvation.timeout=250000 \
                --conf spark.rpc.askTimeout=600 \
                --conf spark.blockManager.port=10025 \
                --conf spark.driver.host=127.0.0.1 \
                --conf spark.driver.blockManager.port=10026 \
                --conf spark.io.compression.codec=lz4 \
                --class com.intel.analytics.bigdl.models.lenet.Train \
                --driver-memory 10G \
                /bin/jars/bigdl-0.13.0-jar-with-dependencies.jar \
                -f /bin/data \
                -b 4 \
                -e 1 | tee spark.local.sgx.log
}

run_spark_resnet_cifar(){
    init_instance spark
    build_spark
    echo -e "${BLUE}occlum run spark${NC}"
    echo -e "${BLUE}logfile=$log${NC}"
    occlum run /usr/lib/jvm/java-11-openjdk-amd64/bin/java \
                -XX:-UseCompressedOops -XX:MaxMetaspaceSize=256m \
                -XX:ActiveProcessorCount=4 \
                -Divy.home="/tmp/.ivy" \
                -Dos.name="Linux" \
                -cp '/bin/conf/:/bin/jars/*'  -Xmx10g org.apache.spark.deploy.SparkSubmit --jars /bin/examples/jars/spark-examples_2.11-2.4.6.jar,/bin/examples/jars/scopt_2.11-3.7.0.jar \
                --master 'local[4]' \
                --conf spark.driver.port=10027 \
                --conf spark.scheduler.maxRegisteredResourcesWaitingTime=5000000 \
                --conf spark.worker.timeout=600 \
                --conf spark.executor.extraClassPath=/bin/jars/bigdl-0.13.0-jar-with-dependencies.jar \
                --conf spark.driver.extraClassPath=/bin/jars/bigdl-0.13.0-jar-with-dependencies.jar \
                --conf spark.starvation.timeout=250000 \
                --conf spark.rpc.askTimeout=600 \
                --conf spark.blockManager.port=10025 \
                --conf spark.driver.host=127.0.0.1 \
                --conf spark.driver.blockManager.port=10026 \
                --conf spark.io.compression.codec=lz4 \
                --class com.intel.analytics.bigdl.models.resnet.TrainCIFAR10 \
                --driver-memory 10G \
                /bin/jars/bigdl-0.13.0-jar-with-dependencies.jar \
                -f /bin/cifar \
                --batchSize 400 --optnet true --depth 20 --classes 10 --shortcutType A --nEpochs 156 \
                --learningRate 0.1 | tee spark.local.sgx.log
}


id=$([ -f "$pid" ] && echo $(wc -l < "$pid") || echo "0")

arg=$1
case "$arg" in
    test)
        run_spark_test
        cd ../
        ;;
    bigdl)
        run_spark_bigdl
        cd ../
        ;;
    cifar)
        run_spark_resnet_cifar
        cd ../
        ;;
    spark)
        init_instance spark
        build_spark
        run_spark
        cd ../
        ;;
esac
