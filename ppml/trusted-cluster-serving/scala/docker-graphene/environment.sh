#!/usr/bin/bash

export MASTER=YOUR_MASTER_IP
export WORKERS=(YOUR_WORKER_IP_1 YOUR_WORKER_IP_2 YOUR_WORKER_IP_3)

export TRUSTED_CLUSTER_SERVING_DOCKER=intelanalytics/analytics-zoo-ppml-trusted-cluster-serving-scala-graphene:0.10-SNAPSHOT

export KEYS_PATH=/opt/analytics-zoo-ppml/keys
export SECURE_PASSWORD_PATH=/opt/analytics-zoo-ppml/password
