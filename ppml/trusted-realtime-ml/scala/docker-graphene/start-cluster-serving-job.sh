#!/bin/bash

set -x

flink_home=$FLINK_HOME
flink_job_manager_ip=$FLINK_JOB_MANAGER_IP
flink_job_manager_port=$FLINK_JOB_MANAGER_REST_PORT
sgx_mode=$SGX_MODE

echo "### Launching Cluster Serving Job ###"

export FLINK_CONF_DIR=${flink_home}/conf && \
    ${flink_home}/bin/flink run \
    -c com.intel.analytics.bigdl.serving.ClusterServing -p 1 -m $flink_job_manager_ip:$flink_job_manager_port \
    /ppml/trusted-realtime-ml/java/work/bigdl-serving-jar-with-dependencies.jar | tee ./cluster-serving-job-${sgx_mode}.log
