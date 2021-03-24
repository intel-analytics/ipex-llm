#!/usr/bin/bash

export MASTER=YOUR_MASTER_IP
export WORKERS=(YOUR_WORKER_IP_1 YOUR_WORKER_IP_2 YOUR_WORKER_IP_3)

export TRUSTED_CLUSTER_SERVING_DOCKER=intelanalytics/analytics-zoo-ppml-trusted-cluster-serving-scala-graphene:0.10-SNAPSHOT

export SOURCE_KEYS_PATH=YOUR_LOCAL_KEYS_PATH
export SOURCE_SECURE_PASSWORD_PATH=YOUR_LOCAL_SECURE_PASSWORD_PATH

export AZ_PPML_PATH=/opt/analytics-zoo-ppml
export KEYS_PATH=$AZ_PPML_PATH/keys
export SECURE_PASSWORD_PATH=$AZ_PPML_PATH/password
