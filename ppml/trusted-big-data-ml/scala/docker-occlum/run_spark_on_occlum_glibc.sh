#!/bin/bash
set -x
export SGX_MEM_SIZE=64GB

BLUE='\033[1;34m'
NC='\033[0m'
occlum_glibc=/opt/occlum/glibc/lib

init_instance() {
    # Init Occlum instance
    cd /opt
    # check if occlum_spark exists
    [[ -d occlum_spark ]] || mkdir occlum_spark
    cd occlum_spark
    occlum init
    new_json="$(jq '.resource_limits.user_space_size = "SGX_MEM_SIZE" |
        .resource_limits.max_num_of_threads = 512 |
        .process.default_heap_size = "512MB" |
        .resource_limits.kernel_space_heap_size="1024MB" |
        .process.default_mmap_size = "28000MB" |
        .entry_points = [ "/usr/lib/jvm/java-11-openjdk-amd64/bin" ] |
        .env.default = [ "LD_LIBRARY_PATH=/usr/lib/jvm/java-11-openjdk-amd64/lib/server:/usr/lib/jvm/java-11-openjdk-amd64/lib:/usr/lib/jvm/java-11-openjdk-amd64/../lib:/lib","SPARK_CONF_DIR=/bin/conf","SPARK_ENV_LOADED=1","PYTHONHASHSEED=0","SPARK_HOME=/bin","SPARK_SCALA_VERSION=2.12","SPARK_JARS_DIR=/bin/jars","LAUNCH_CLASSPATH=/bin/jars/*",""]' Occlum.json)" && \
    echo "${new_json}" > Occlum.json
    echo "SGX_MEM_SIZE ${SGX_MEM_SIZE}"
    if [[ -z "$SGX_MEM_SIZE" ]]; then
        sed -i "s/SGX_MEM_SIZE/20GB/g" Occlum.json
    else
        sed -i "s/SGX_MEM_SIZE/${SGX_MEM_SIZE}/g" Occlum.json
    fi
}

build_spark() {
    # Copy JVM and class file into Occlum instance and build
    cd /opt/occlum_spark
    mkdir -p image/usr/lib/jvm
    cp -r /usr/lib/jvm/java-11-openjdk-amd64 image/usr/lib/jvm
    cp -rf /etc/java-11-openjdk image/etc/
    # Copy libs
    cp /lib/x86_64-linux-gnu/libz.so.1 image/lib
    cp /lib/x86_64-linux-gnu/libz.so.1 image/$occlum_glibc
    cp /lib/x86_64-linux-gnu/libtinfo.so.5 image/$occlum_glibc
    cp /lib/x86_64-linux-gnu/libnss*.so.2 image/$occlum_glibc
    cp /lib/x86_64-linux-gnu/libresolv.so.2 image/$occlum_glibc
    cp $occlum_glibc/libdl.so.2 image/$occlum_glibc
    cp $occlum_glibc/librt.so.1 image/$occlum_glibc
    cp $occlum_glibc/libm.so.6 image/$occlum_glibc
    # Copy libhadoop
    cp /opt/libhadoop.so image/lib
    # Prepare Spark
    mkdir -p image/opt/spark
    cp -rf $SPARK_HOME/* image/opt/spark/
    # Copy etc files
    cp -rf /etc/hosts image/etc/
    echo "$HOST_IP occlum-node" >> image/etc/hosts
    # cat image/etc/hosts

    cp -rf /etc/hostname image/etc/
    cp -rf /etc/ssl image/etc/
    cp -rf /etc/passwd image/etc/
    cp -rf /etc/group image/etc/
    cp -rf /etc/nsswitch.conf image/etc/

    # Prepare BigDL
    mkdir -p image/bin/jars
    cp -f $BIGDL_HOME/jars/* image/bin/jars
    occlum build
}

run_spark_pi() {
    init_instance spark
    build_spark
    echo -e "${BLUE}occlum run spark Pi${NC}"
    occlum run /usr/lib/jvm/java-11-openjdk-amd64/bin/java \
                -XX:-UseCompressedOops -XX:MaxMetaspaceSize=256m \
                -XX:ActiveProcessorCount=4 \
                -Divy.home="/tmp/.ivy" \
                -Dos.name="Linux" \
                -cp "$SPARK_HOME/conf/:$SPARK_HOME/jars/*" \
                -Xmx10g org.apache.spark.deploy.SparkSubmit \
                --jars $SPARK_HOME/examples/jars/spark-examples_2.12-3.1.2.jar,$SPARK_HOME/examples/jars/scopt_2.12-3.7.1.jar \
                --class org.apache.spark.examples.SparkPi spark-internal
}

run_spark_unittest() {
    init_instance spark
    build_spark
    echo -e "${BLUE}occlum run spark unit test ${NC}"
    run_spark_unittest_only
}

run_spark_unittest_only() {
    cd /opt/occlum_spark
    mkdir -p olog
    echo -e "${BLUE}occlum run spark unit test only ${NC}"
    for suite in `cat /opt/sqlSuites`
    do occlum run /usr/lib/jvm/java-11-openjdk-amd64/bin/java -Xmx24g \
                -Divy.home="/tmp/.ivy" \
                -Dos.name="Linux" \
		-Djdk.lang.Process.launchMechanism=posix_spawn \
		-XX:MaxMetaspaceSize=256m \
	        -Dspark.testing=true \
	        -Dspark.test.home=/ppml/trusted-big-data-ml/work/spark-branch-3.1.2 \
	        -Dspark.sql.warehouse.dir=hdfs://localhost:9000/111-spark-warehouse \
	        -Dspark.python.use.daemon=false \
	        -Dspark.python.worker.reuse=false \
	        -Dspark.driver.host=192.168.0.111 \
	        -cp "$SPARK_HOME/conf/:$SPARK_HOME/jars/*:$SPARK_HOME/test-jars/*"  \
	        org.scalatest.tools.Runner \
	        -s ${suite} \
	        -fF /host/olog/${suite}.txt
    done
}

run_spark_lenet_mnist(){
    init_instance spark
    build_spark
    echo -e "${BLUE}occlum run BigDL lenet mnist{NC}"
    echo -e "${BLUE}logfile=$log${NC}"
    occlum run /usr/lib/jvm/java-11-openjdk-amd64/bin/java \
                -XX:-UseCompressedOops -XX:MaxMetaspaceSize=256m \
                -XX:ActiveProcessorCount=4 \
                -Divy.home="/tmp/.ivy" \
                -Dos.name="Linux" \
                -cp "$SPARK_HOME/conf/:$SPARK_HOME/jars/*:/bin/jars/*" \
                -Xmx10g org.apache.spark.deploy.SparkSubmit \
                --master 'local[4]' \
                --conf spark.driver.port=10027 \
                --conf spark.scheduler.maxRegisteredResourcesWaitingTime=5000000 \
                --conf spark.worker.timeout=600 \
                --conf spark.starvation.timeout=250000 \
                --conf spark.rpc.askTimeout=600 \
                --conf spark.blockManager.port=10025 \
                --conf spark.driver.host=127.0.0.1 \
                --conf spark.driver.blockManager.port=10026 \
                --conf spark.io.compression.codec=lz4 \
                --class com.intel.analytics.bigdl.dllib.models.lenet.Train \
                --driver-memory 10G \
                /bin/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar \
                -f /host/data \
                $* | tee spark.local.sgx.log
}

run_spark_resnet_cifar(){
    init_instance spark
    build_spark
    echo -e "${BLUE}occlum run BigDL Resnet Cifar10${NC}"
    occlum run /usr/lib/jvm/java-11-openjdk-amd64/bin/java \
                -XX:-UseCompressedOops -XX:MaxMetaspaceSize=256m \
                -XX:ActiveProcessorCount=4 \
                -Divy.home="/tmp/.ivy" \
                -Dos.name="Linux" \
                -cp "$SPARK_HOME/conf/:$SPARK_HOME/jars/*:/bin/jars/*" \
                -Xmx10g org.apache.spark.deploy.SparkSubmit \
                --master 'local[4]' \
                --conf spark.driver.port=10027 \
                --conf spark.scheduler.maxRegisteredResourcesWaitingTime=5000000 \
                --conf spark.worker.timeout=600 \
                --conf spark.starvation.timeout=250000 \
                --conf spark.rpc.askTimeout=600 \
                --conf spark.blockManager.port=10025 \
                --conf spark.driver.host=127.0.0.1 \
                --conf spark.driver.blockManager.port=10026 \
                --conf spark.io.compression.codec=lz4 \
                --class com.intel.analytics.bigdl.dllib.models.resnet.TrainCIFAR10 \
                --driver-memory 10G \
                /bin/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar \
                -f /host/data \
                $* | tee spark.local.sgx.log
}

run_spark_tpch(){
    init_instance spark
    build_spark
    echo -e "${BLUE}occlum run BigDL spark tpch${NC}"
    occlum run /usr/lib/jvm/java-11-openjdk-amd64/bin/java \
                -XX:-UseCompressedOops -XX:MaxMetaspaceSize=256m \
                -XX:ActiveProcessorCount=4 \
                -Divy.home="/tmp/.ivy" \
                -Dos.name="Linux" \
                -cp "$SPARK_HOME/conf/:$SPARK_HOME/jars/*:/bin/jars/*" \
                -Xmx32g org.apache.spark.deploy.SparkSubmit \
                --master 'local[4]' \
                --conf spark.driver.port=54321 \
                --conf spark.driver.memory=8g \
                --conf spark.driver.blockManager.port=10026 \
                --conf spark.blockManager.port=10025 \
                --conf spark.scheduler.maxRegisteredResourcesWaitingTime=5000000 \
                --conf spark.worker.timeout=600 \
                --conf spark.python.use.daemon=false \
                --conf spark.python.worker.reuse=false \
                --conf spark.network.timeout=10000000 \
                --conf spark.starvation.timeout=250000 \
                --conf spark.rpc.askTimeout=600 \
                --conf spark.sql.autoBroadcastJoinThreshold=-1 \
                --conf spark.io.compression.codec=lz4 \
                --conf spark.sql.shuffle.partitions=8 \
                --conf spark.speculation=false \
                --conf spark.executor.heartbeatInterval=10000000 \
                --conf spark.executor.instances=4 \
                --executor-cores 8 \
                --total-executor-cores 32 \
                --executor-memory 8G \
                --class main.scala.TpchQuery \
                --verbose \
                /bin/jars/spark-tpc-h-queries_2.12-1.0.jar \
                /host/data /host/data/output
}

id=$([ -f "$pid" ] && echo $(wc -l < "$pid") || echo "0")

arg=$1
case "$arg" in
    init)
        init_instance
        build_spark
        ;;
    pi)
        run_spark_pi
        cd ../
        ;;
    lenet)
        run_spark_lenet_mnist
        cd ../
        ;;
    ut)
        run_spark_unittest
        cd ../
        ;;
    ut_Only)
        run_spark_unittest_only
        cd ../
        ;;
    resnet)
        run_spark_resnet_cifar
        cd ../
        ;;
    tpch)
        run_spark_tpch
        cd ../
        ;;
esac

