#!/bin/bash
set -ex

# Check whether there is a passwd entry for the container UID
myuid=$(id -u)
mygid=$(id -g)
# turn off -e for getent because it will return error code in anonymous uid case
set +e
uidentry=$(getent passwd $myuid)
set -e

# To use sgx_sign in k8s env
source /opt/intel/sgxsdk/environment
# If there is no passwd entry for the container UID, attempt to create one
if [ -z "$uidentry" ] ; then
    if [ -w /etc/passwd ] ; then
        echo "$myuid:x:$myuid:$mygid:anonymous uid:$SPARK_HOME:/bin/false" >> /etc/passwd
    else
        echo "Container ENTRYPOINT failed to add passwd entry for anonymous UID"
    fi
fi

#check glic ENV MALLOC_ARENA_MAX for k8s
if [[ -z "$MALLOC_ARENA_MAX" ]]; then
    echo "No MALLOC_ARENA_MAX specified, set to 1."
    export MALLOC_ARENA_MAX=1
fi

# check occlum log level for k8s
if [[ -z "$ENABLE_SGX_DEBUG" ]]; then
    echo "No ENABLE_SGX_DEBUG specified, set to off."
    export ENABLE_SGX_DEBUG=false
fi
export OCCLUM_LOG_LEVEL=off
if [[ -z "$SGX_LOG_LEVEL" ]]; then
    echo "No SGX_LOG_LEVEL specified, set to off."
else
    echo "Set SGX_LOG_LEVEL to $SGX_LOG_LEVEL"
    if [[ $SGX_LOG_LEVEL == "debug" ]] || [[ $SGX_LOG_LEVEL == "trace" ]]; then
        export ENABLE_SGX_DEBUG=true
        export OCCLUM_LOG_LEVEL=$SGX_LOG_LEVEL
    fi
fi

usage() {
  echo "Usage: $0 [-m --mode <controller|worker>] [-h --help]"
  echo "-h: Print help message."
  echo "Controller mode reads the following env:"
  echo "CONTROLLER_HOST (default: localhost)."
  echo "CONTROLLER_PORT (default: 21001)."
  echo "API_HOST (default: localhost)."
  echo "API_PORT (default: 8000)."
  echo "Worker mode reads the following env:"
  echo "CONTROLLER_HOST (default: localhost)."
  echo "CONTROLLER_PORT (default: 21001)."
  echo "WORKER_HOST (default: localhost)."
  echo "WORKER_PORT (default: 21002)."
  echo "MODEL_PATH (default: empty)."
  echo "ENABLE_ATTESTATION_API (default: empty)."
  exit 1
}

# Default values
controller_host="localhost"
controller_port="21001"
api_host="localhost"
api_port="8000"
worker_host="localhost"
worker_port="21002"
model_path=""
mode=""
omp_num_threads=""
attest_flag=""

# Update rootCA config if needed
update-ca-certificates

# Remember the value of `OMP_NUM_THREADS`:
if [[ -n "${OMP_NUM_THREADS}" ]]; then
  omp_num_threads="${OMP_NUM_THREADS}"
fi

# We do not have any arguments, just run bash
if [ "$#" == 0 ]; then
  echo "[INFO] no command is passed in"
  echo "[INFO] enter pass-through mode"
  exec /sbin/tini -s -- "bash"
else
  # Parse command-line options
  options=$(getopt -o "m:h" --long "mode:,help" -n "$0" -- "$@")
  if [ $? != 0 ]; then
    usage
  fi
  eval set -- "$options"

  while true; do
    case "$1" in
      -m|--mode)
        mode="$2"
        [[ $mode == "controller" || $mode == "worker" ]] || usage
        shift 2
        break
        ;;
      -h|--help)
        usage
        ;;
      --)
        exec /sbin/tini -s "$@"
        ;;
      *)
        usage
        ;;
    esac
  done

  if [[ -n $CONTROLLER_HOST ]]; then
    controller_host=$CONTROLLER_HOST
  fi

  if [[ -n $CONTROLLER_PORT ]]; then
    controller_port=$CONTROLLER_PORT
  fi

  if [[ -n $API_HOST ]]; then
    api_host=$API_HOST
  fi

  if [[ -n $API_PORT ]]; then
    api_port=$API_PORT
  fi

  if [[ -n $WORKER_HOST ]]; then
    worker_host=$WORKER_HOST
  fi

  if [[ -n $WORKER_PORT ]]; then
    worker_port=$WORKER_PORT
  fi

  if [[ -n $MODEL_PATH ]]; then
    model_path=$MODEL_PATH
  fi

  if [[ $ENABLE_ATTESTATION_API = "true" ]]; then
    attest_flag="--attest"
  fi

  controller_address="http://$controller_host:$controller_port"
  # Execute logic based on options
  if [[ $mode == "controller" ]]; then
    # init Occlum
    /opt/run_llm_on_occlum_glibc.sh init
    # Logic for controller mode
    # Boot Controller
    # TODO: add dispatch-method
    api_address="http://$api_host:$api_port"
    echo "Controller address: $controller_address"
    echo "OpenAI API address: $api_address"
    cd /opt/occlum_spark
    occlum start
    occlum exec /bin/python3 -m fastchat.serve.controller --host $controller_host --port $controller_port $attest_flag &
    # Boot openai api server
    occlum exec /bin/python3 -m fastchat.serve.openai_api_server --host $api_host --port $api_port --controller-address $controller_address $attest_flag
  elif [[ $mode == "worker" ]]; then
    # init Occlum
    /opt/run_llm_on_occlum_glibc.sh init
    # Logic for non-controller(worker) mode
    worker_address="http://$worker_host:$worker_port"
    # Apply optimizations from bigdl-nano
    #source bigdl-nano-init -t
    # First check if user have set OMP_NUM_THREADS by themselves
    if [[ -n "${omp_num_threads}" ]]; then
      echo "Setting OMP_NUM_THREADS to its original value: $omp_num_threads"
      export OMP_NUM_THREADS=$omp_num_threads
    else
      # Use default settings
      export OMP_NUM_THREADS=16
    fi
    if [[ -z "${model_path}" ]]; then
          echo "Please set env MODEL_PATH used for worker"
          usage
    fi
    echo "Worker address: $worker_address"
    echo "Controller address: $controller_address"
    cd /opt/occlum_spark
    occlum start
    occlum exec /bin/python3 -m fastchat.serve.model_worker --model-path $model_path --device cpu --host $worker_host --port $worker_port --worker-address $worker_address --controller-address $controller_address $attest_flag
  fi
fi
