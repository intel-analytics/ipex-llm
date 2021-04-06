#!/bin/bash
set -e

BLUE='\033[1;34m'
NC='\033[0m'
conf_dir=/opt/conf

id=$([ -f "$pid" ] && echo $(wc -l < "$pid") || echo "0")
FLINK_LOG_PREFIX="/host/flink--$postfix-${id}"
log="${FLINK_LOG_PREFIX}.log"

core_num=$CORE_NUM
job_manager_host=$FLINK_JOB_MANAGER_IP
job_manager_rest_port=$FLINK_JOB_MANAGER_REST_PORT
job_manager_rpc_port=$FLINK_JOB_MANAGER_RPC_PORT
secure_password=`openssl rsautl -inkey /opt/password/key.txt -decrypt </opt/password/output.bin`
flink_home=$FLINK_HOME
flink_version=$FLINK_VERSION

run_taskmanager() {
    # enter occlum image
    cd /opt/flink

    #if conf_dir exists, use the new configurations.
    if [ -d $conf_dir  ];then
        cp -r $conf_dir/* image/opt/conf/
        occlum build
    fi

    echo -e "${BLUE}occlum run JVM taskmanager${NC}"
    echo -e "${BLUE}logfile=$log${NC}"
    # start task manager in occlum
    occlum run /usr/lib/jvm/java-11-openjdk-amd64/bin/java \
    -XX:+UseG1GC -Xmx25g -Xms6g -XX:MaxDirectMemorySize=3g -XX:MaxMetaspaceSize=4g \
    -Dos.name=Linux \
    -XX:ActiveProcessorCount=${core_num} \
    -Dlog.file=$log \
    -Dlog4j.configuration=file:/opt/conf/log4j.properties \
    -Dlogback.configurationFile=file:/opt/conf/logback.xml \
    -classpath /bin/lib/flink-table-blink_2.11-1.10.1.jar:/bin/lib/flink-table_2.11-1.10.1.jar:/bin/lib/log4j-1.2.17.jar:/bin/lib/slf4j-log4j12-1.7.15.jar:/bin/lib/flink-dist_2.11-1.10.1.jar org.apache.flink.runtime.taskexecutor.TaskManagerRunner \
    -Dorg.apache.flink.shaded.netty4.io.netty.tryReflectionSetAccessible=true \
    -Dorg.apache.flink.shaded.netty4.io.netty.eventLoopThreads=${core_num} \
    -Dcom.intel.analytics.zoo.shaded.io.netty.tryReflectionSetAccessible=true \
    --configDir /opt/conf \
    -D rest.bind-address=${job_manager_host} \
    -D rest.bind-port=${job_manager_rest_port} \
    -D jobmanager.rpc.address=${job_manager_host} \
    -D jobmanager.rpc.port=${job_manager_rpc_port} \
    -D jobmanager.heap.size=5g \
    -D security.ssl.internal.enabled=true \
    -D security.ssl.internal.keystore=/opt/keys/keystore.pkcs12 \
    -D security.ssl.internal.truststore=/opt/keys/keystore.pkcs12 \
    -D security.ssl.internal.keystore-password=${secure_password} \
    -D security.ssl.internal.truststore-password=${secure_password} \
    -D security.ssl.internal.key-password=${secure_password} \
    -D taskmanager.memory.framework.off-heap.size=256mb \
    -D taskmanager.memory.network.max=1024mb \
    -D taskmanager.memory.network.min=1024mb \
    -D taskmanager.memory.framework.heap.size=256mb \
    -D taskmanager.memory.managed.size=8g \
    -D taskmanager.cpu.cores=1.0 \
    -D taskmanager.memory.task.heap.size=10gb \
    -D taskmanager.memory.task.off-heap.size=1024mb &
}

run_taskmanager
