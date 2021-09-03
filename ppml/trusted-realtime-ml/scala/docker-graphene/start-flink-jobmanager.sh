#!/bin/bash

set -x

core_num=$CORE_NUM
job_manager_host=$FLINK_JOB_MANAGER_IP
job_manager_rest_port=$FLINK_JOB_MANAGER_REST_PORT
job_manager_rpc_port=$FLINK_JOB_MANAGER_RPC_PORT
secure_password=`openssl rsautl -inkey /ppml/trusted-realtime-ml/java/work/password/key.txt -decrypt </ppml/trusted-realtime-ml/java/work/password/output.bin`
flink_home=$FLINK_HOME
flink_version=$FLINK_VERSION
blob_server_port=$FLINK_BLOB_SERVER_PORT
blob_server_port=$FLINK_BLOB_SERVER_PORT
xmx_size=$XMX_SIZE
sgx_mode=$SGX_MODE

if [[ $sgx_mode == "sgx" || $sgx_mode == "SGX" ]];then cmd_prefix="graphene-sgx ./"; fi

echo "### Launching Flink Jobmanager ###"

eval ${cmd_prefix}bash -c \" /opt/jdk8/bin/java \
    -Xms5g \
    -Xmx${xmx_size} \
    -XX:ActiveProcessorCount=${core_num} \
    -Djdk.tls.client.protocols="TLSv1,TLSv1.1,TLSv1.2" \
    -Dorg.apache.flink.shaded.netty4.io.netty.tryReflectionSetAccessible=true \
    -Dorg.apache.flink.shaded.netty4.io.netty.eventLoopThreads=${core_num} \
    -Dcom.intel.analytics.zoo.shaded.io.netty.tryReflectionSetAccessible=true \
    -Dlog.file=${flink_home}/log/flink-standalonesession-1-${sgx_mode}.log \
    -Dlog4j.configurationFile=file:${flink_home}/conf/log4j.properties \
    -Dlogback.configurationFile=file:${flink_home}/conf/logback.xml \
    -classpath ${flink_home}/lib/flink-csv-${flink_version}.jar:${flink_home}/lib/flink-dist_2.11-${flink_version}.jar:${flink_home}/lib/flink-json-${flink_version}.jar:${flink_home}/lib/flink-shaded-zookeeper-3.4.14.jar:${flink_home}/lib/flink-table_2.11-${flink_version}.jar:${flink_home}/lib/flink-table-blink_2.11-${flink_version}.jar:${flink_home}/lib/log4j-1.2-api-2.12.1.jar:${flink_home}/lib/log4j-api-2.12.1.jar:${flink_home}/lib/log4j-core-2.12.1.jar:${flink_home}/lib/log4j-slf4j-impl-2.12.1.jar::: org.apache.flink.runtime.entrypoint.StandaloneSessionClusterEntrypoint \
    --configDir ${flink_home}/conf \
    -D rest.bind-address=0.0.0.0 \
    -D rest.bind-port=${job_manager_rest_port} \
    -D blob.server.port=${blob_server_port} \
    -D jobmanager.rpc.address=${job_manager_host} \
    -D jobmanager.rpc.port=${job_manager_rpc_port} \
    -D jobmanager.heap.size=5g \
    -D security.ssl.internal.enabled=true \
    -D security.ssl.internal.keystore=/ppml/trusted-realtime-ml/java/work/keys/keystore.pkcs12 \
    -D security.ssl.internal.truststore=/ppml/trusted-realtime-ml/java/work/keys/keystore.pkcs12 \
    -D security.ssl.internal.keystore-password=${secure_password} \
    -D security.ssl.internal.truststore-password=${secure_password} \
    -D security.ssl.internal.key-password=${secure_password} \
    --executionMode cluster \" | tee ./flink-jobmanager-${sgx_mode}.log
