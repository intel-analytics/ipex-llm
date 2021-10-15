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
  echo "failed,Please install numactl package to activate PERFORMANCE_MODE"
  exit 1
fi

if ht_enabled; then
  echo "failed,Please turn off Hyperthreading to activate PERFORMANCE_MODE"
  exit 1
fi

_WORKER_CORE_NUM_LESS_THAN=24
_MODE=$1 # start or stop
_WORKER_PER_SOCKET=$2  # worker num on each numa node

if [ "${_MODE}" != "start" ] && [ "${_MODE}" != "stop" ]; then
  echo "failed,mode should be start or stop."
  exit 1
fi

TOTAL_CORE_NUM=`nproc`
if ht_enabled; then
  TOTAL_CORE_NUM=$((TOTAL_CORE_NUM / 2))
fi

if [ $TOTAL_CORE_NUM -lt $_WORKER_CORE_NUM_LESS_THAN ] && [ -z "${_WORKER_PER_SOCKET}" ]; then
  # use local mode
  echo "local[*]"
  exit 1
fi

. "${SPARK_HOME}/sbin/spark-config.sh"

. "${SPARK_HOME}/bin/load-spark-env.sh"

if [ -z "${SPARK_MASTER_HOST}" ]; then
  export SPARK_MASTER_HOST=`hostname`
fi

if [ -z "${SPARK_MASTER_PORT}" ]; then
  export SPARK_MASTER_PORT=7077
fi

if [ -z "${SPARK_MASTER_WEBUI_PORT}" ]; then
  export SPARK_MASTER_WEBUI_PORT=8080
fi

grep_port=`netstat -tlpn | awk '{print $4}' | grep "\b$SPARK_MASTER_PORT\b"`
if [ "${_MODE}" == "start" ] && [ -n "$grep_port" ]; then
  echo "failed,Spark master port $SPARK_MASTER_PORT is in use"
  exit 1
fi

# NOTE: This exact class name is matched downstream by SparkSubmit.
# Any changes need to be reflected there.
CLASS="org.apache.spark.deploy.worker.Worker"

MASTER="spark://$SPARK_MASTER_HOST:$SPARK_MASTER_PORT"

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

  $NUMACTL "${SPARK_HOME}/sbin"/spark-daemon.sh start $CLASS $WORKER_NUM \
     --webui-port "$WEBUI_PORT" $PORT_FLAG $PORT_NUM $MASTER "$@"
}

function stop_instance {
  WORKER_NUM=$1
  "${SPARK_HOME}/sbin"/spark-daemon.sh stop org.apache.spark.deploy.worker.Worker "$WORKER_NUM"
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
  echo "failed,Not enough memory for cluster serving performance mode"
  exit 1
fi

# Start or stop master node
"${SPARK_HOME}/sbin"/spark-daemon.sh "${_MODE}" org.apache.spark.deploy.master.Master spark-master --ip $SPARK_MASTER_HOST --port $SPARK_MASTER_PORT --webui-port $SPARK_MASTER_WEBUI_PORT

_WORKER_NAME_NO=1

# Load NUMA configurations line-by-line and set `numactl` options
for nnode in ${_NUMA_HARDWARE_INFO[@]}; do
  if [[ ${nnode} =~ ^node\ ([0-9]+)\ cpus:\ (.+)$ ]]; then
    _NUMA_NO=${BASH_REMATCH[1]}
    IFS=' ' _NUMA_CPUS=(${BASH_REMATCH[2]})
    _LENGTH=${#_NUMA_CPUS[@]}
    if ht_enabled; then _LENGTH=$((_LENGTH / 2)); fi
    if [[ -z "${_WORKER_PER_SOCKET}" ]]; then
      # calculate worker num on this numa node, 12 ~ 23 core/worker
      _WORKER_PER_SOCKET=$((_LENGTH / $((_WORKER_CORE_NUM_LESS_THAN / 2))))
    fi
    if [[ $_WORKER_PER_SOCKET -eq 0 ]]; then
      _WORKER_PER_SOCKET=1
    fi
    if [[ $_WORKER_PER_SOCKET -gt $_NUMA_MEM ]]; then
      _WORKER_PER_SOCKET=$_NUMA_MEM
    fi
    _LENGTH=$((_LENGTH / _WORKER_PER_SOCKET))

    for ((i = 0; i < $((_WORKER_PER_SOCKET)); i ++)); do
      if [ "${_MODE}" == "start" ]; then
        core_start=$(( i * _LENGTH ))
        _NUMACTL="numactl -m ${_NUMA_NO} -C $(join_by , ${_NUMA_CPUS[@]:${core_start}:${_LENGTH}})"
        echo ${_NUMACTL}

        # Launch a worker with numactl
        export SPARK_WORKER_CORES=${_LENGTH} # core num per worker
        export SPARK_WORKER_MEMORY="$((_NUMA_MEM / _WORKER_PER_SOCKET))g"
        start_instance "$_NUMACTL" "$_WORKER_NAME_NO"
      else
        stop_instance "$_WORKER_NAME_NO"
      fi
      _WORKER_NAME_NO=$((_WORKER_NAME_NO + 1))
    done
  fi
done

if [ "${_MODE}" == "start" ]; then
  # master, executor cores, executor num, total executor cores, driver memory
  echo "$MASTER,$_LENGTH,$((_WORKER_NAME_NO - 1)),$((_LENGTH * $((_WORKER_NAME_NO - 1)))),${_MEMORY_FOR_DRIVER}g"
fi
