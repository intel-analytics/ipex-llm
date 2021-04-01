#!/bin/bash

set -x

flink_home=$FLINK_HOME
flink_job_manager_ip=$FLINK_JOB_MANAGER_IP
flink_job_manager_port=$FLINK_JOB_MANAGER_REST_PORT


echo "### Launching Cluster Serving Job ###"

export FLINK_CONF_DIR=${flink_home}/conf && \
    ${flink_home}/bin/flink run \
    -c com.intel.analytics.zoo.serving.ClusterServing -p 1 -m $flink_job_manager_ip:$flink_job_manager_port \
    /opt/analytics-zoo/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-serving.jar | tee ./cluster-serving-job-sgx.log
