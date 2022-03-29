#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_WORKLOAD_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/workload

set -e
echo "# Start testing"
start=$(date "+%s")

function runpythonfiles(){
  for file in "ls $1"
  do
    if [-d 1"/"$file]
    then
      runpythonfiles($1"/"$file)
    else
      python -m $1"/"$file
    fi
    done
}

runpythonfiles ${NANO_WORKLOAD_DIR}

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-nano tests finished"
echo "Time used:$time seconds"

