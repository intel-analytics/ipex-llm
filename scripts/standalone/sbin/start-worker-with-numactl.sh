#!/usr/bin/env bash

function ht_enabled {
  ret=`lscpu |grep "Thread(s) per core"|awk '{print $4}'`
  if [ $ret -eq 1 ]; then
    false
  else
    true
  fi
}

# check if we can start performance mode
if [ -z "${SPARK_HOME}" ]; then
  echo "failed,Please set SPARK_HOME environment variable"
  exit 1
fi

if ! type "numactl" > /dev/null 2>&1; then
  echo "failed, please install numactl package"
  exit 1
fi

MASTER=$1
_TOTAL_WORKERS=$SPARK_WORKER_INSTANCES
#_WORKER_PER_SOCKET=$2  # worker num on each numa node

TOTAL_CORE_NUM=`nproc`
if ht_enabled; then
  TOTAL_CORE_NUM=$((TOTAL_CORE_NUM / 2))
fi

. "${ZOO_STANDALONE_HOME}/sbin/spark-config.sh"

. "${SPARK_HOME}/bin/load-spark-env.sh"

# NOTE: This exact class name is matched downstream by SparkSubmit.
# Any changes need to be reflected there.
CLASS="org.apache.spark.deploy.worker.Worker"

# Determine desired worker port
if [ "$SPARK_WORKER_WEBUI_PORT" = "" ]; then
  SPARK_WORKER_WEBUI_PORT=8081
fi

# Start up the appropriate number of workers on this machine.
# quick local function to start a worker
function start_instance {
  NUMACTL=$1
  WORKER_NUM=$2
  shift
  shift

  if [ "$SPARK_WORKER_PORT" = "" ]; then
    PORT_FLAG=
    PORT_NUM=
  else
    PORT_FLAG="--port"
    PORT_NUM=$(( $SPARK_WORKER_PORT + $WORKER_NUM - 1 ))
  fi
  WEBUI_PORT=$(( $SPARK_WORKER_WEBUI_PORT + $WORKER_NUM - 1 ))

  $NUMACTL "${ZOO_STANDALONE_HOME}/sbin"/spark-daemon.sh start $CLASS $WORKER_NUM \
     --webui-port "$WEBUI_PORT" $PORT_FLAG $PORT_NUM $MASTER "$@"
}

# Join an input array by a given separator
function join_by() {
  local IFS="$1"
  shift
  echo "$*"
}

# Compute memory size for each NUMA node
IFS=$'\n'; _NUMA_HARDWARE_INFO=(`numactl --hardware`)
_NUMA_NODE_NUM=`echo ${_NUMA_HARDWARE_INFO[0]} | sed -e "s/^available: \([0-9]*\) nodes .*$/\1/"`
_TOTAL_MEM=`grep MemTotal /proc/meminfo | awk '{print $2}'`
# Memory size of each NUMA node = (Total memory size - 1g) / Num of NUMA nodes
_1G=1048576
_MEMORY_FOR_DRIVER=2  # reserve 2g memory for the driver
_NUMA_MEM=$((((_TOTAL_MEM - _1G - (_1G * _MEMORY_FOR_DRIVER)) / _1G) / $_NUMA_NODE_NUM))
if [[ $_NUMA_MEM -le 0 ]]; then
  echo "failed,Not enough memory for numa binding"
  exit 1
fi

if [[ $((_TOTAL_WORKERS % _NUMA_NODE_NUM)) -eq 0 ]]; then
	_WORKER_PER_SOCKET=$((_TOTAL_WORKERS / _NUMA_NODE_NUM))
else
  echo "failed, SPARK_WORKER_INSTANCES should be a multiple of the number of numa nodes. Got SPARK_WORKER_INSTANCES: ${_TOTAL_WORKERS}, and numa node number: ${_NUMA_NODE_NUM}"
  exit 1
fi

_WORKER_NAME_NO=1

# Load NUMA configurations line-by-line and set `numactl` options
for nnode in ${_NUMA_HARDWARE_INFO[@]}; do
  if [[ ${nnode} =~ ^node\ ([0-9]+)\ cpus:\ (.+)$ ]]; then
    _NUMA_NO=${BASH_REMATCH[1]}
    IFS=' ' _NUMA_CPUS=(${BASH_REMATCH[2]})
    _LENGTH=${#_NUMA_CPUS[@]}
    if ht_enabled; then _LENGTH=$((_LENGTH / 2)); fi

    _PER_WORKER_LENGTH=$((_LENGTH / _WORKER_PER_SOCKET))

    for ((i = 0; i < $((_WORKER_PER_SOCKET)); i++)); do
      core_start=$(( i * _PER_WORKER_LENGTH ))
      _NUMACTL="numactl -m ${_NUMA_NO} -C $(join_by , ${_NUMA_CPUS[@]:${core_start}:${_PER_WORKER_LENGTH}})"
      if ht_enabled; then _NUMACTL="$_NUMACTL,$(join_by , ${_NUMA_CPUS[@]:$((core_start + _LENGTH)):${_PER_WORKER_LENGTH}})"; fi
      echo ${_NUMACTL}

      # Launch a worker with numactl
      export SPARK_WORKER_CORES=${_PER_WORKER_LENGTH} # core num per worker
      export SPARK_WORKER_MEMORY="$((_NUMA_MEM / _WORKER_PER_SOCKET))g"
      start_instance "$_NUMACTL" "$_WORKER_NAME_NO"
      _WORKER_NAME_NO=$((_WORKER_NAME_NO + 1))
    done
  fi
done

