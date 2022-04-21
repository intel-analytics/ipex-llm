#!/bin/bash
IMAGE_TAG="intelanalytics/friesian-serving:0.1.0"
CLUSTER_NAME=""
SERVER_PASS=""
RANKING_SERVER_LIST=('127.0.0.1')
RECALL_SERVER_LIST=('127.0.0.1')
FEATURE_SERVER_LIST=('127.0.0.1')
FEATURE_RECALL_SERVER_LIST=('127.0.0.1')
RECOMMENDER_SERVER_LIST=('127.0.0.1')

# Use nfs volume
VOLUME_TARGET="/opt/work/nfs"
VOLUME_DEVICE="/mnt/myshareddir"
VOLUME_ADDR="172.168.3.115"
VOLUME_CONF="type=volume,source=nfsvolume,target=$VOLUME_TARGET,volume-driver=local,volume-opt=type=nfs,volume-opt=device=:$VOLUME_DEVICE,volume-opt=o=addr=$VOLUME_ADDR"

# Use bind mounts
# VOLUME_TARGET="/opt/work/data"
# VOLUME_SOURCE="$(pwd)/data"
# VOLUME_CONF="type=bind,source=$VOLUME_SOURCE,target=$VOLUME_TARGET"

RANKING_INSTANCE_PER_NODE=1
RECOMMENDER_INSTANCE_PER_NODE=1
RECALL_INSTANCE_PER_NODE=1

OMP_NUM_THREADS=1
