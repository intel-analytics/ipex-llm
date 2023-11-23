#!/bin/bash

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

# Acquire correct core_nums if using cpuset-cpus, return -1 if file not exist
calculate_total_cores() {
  local cpuset_file="/sys/fs/cgroup/cpuset/cpuset.cpus"

  if [[ -f "$cpuset_file" ]]; then
    local cpuset_cpus=$(cat "$cpuset_file")
    cpuset_cpus=$(echo "${cpuset_cpus}" | tr -d '\n')

    local total_cores=0
    IFS=',' read -ra cpu_list <<< "$cpuset_cpus"
    for cpu in "${cpu_list[@]}"; do
      if [[ $cpu =~ - ]]; then
        # Range of CPUs
        local start_cpu=$(echo "$cpu" | cut -d'-' -f1)
        local end_cpu=$(echo "$cpu" | cut -d'-' -f2)
        local range_cores=$((end_cpu - start_cpu + 1))
        total_cores=$((total_cores + range_cores))
      else
        # Single CPU
        total_cores=$((total_cores + 1))
      fi
    done

    echo $total_cores
    return
  fi
  # Kubernetes core-binding will use this file
  cpuset_file="/sys/fs/cgroup/cpuset.cpus"
  if [[ -f "$cpuset_file" ]]; then
    local cpuset_cpus=$(cat "$cpuset_file")
    cpuset_cpus=$(echo "${cpuset_cpus}" | tr -d '\n')

    local total_cores=0
    IFS=',' read -ra cpu_list <<< "$cpuset_cpus"
    for cpu in "${cpu_list[@]}"; do
      if [[ $cpu =~ - ]]; then
        # Range of CPUs
        local start_cpu=$(echo "$cpu" | cut -d'-' -f1)
        local end_cpu=$(echo "$cpu" | cut -d'-' -f2)
        local range_cores=$((end_cpu - start_cpu + 1))
        total_cores=$((total_cores + range_cores))
      else
        # Single CPU
        total_cores=$((total_cores + 1))
      fi
    done

    echo $total_cores
    return
  else
    echo -1
    return
  fi
}

# Attestation
if [ -z "$ATTESTATION" ]; then
  echo "[INFO] Attestation is disabled!"
  ATTESTATION="false"
fi

if [ "$ATTESTATION" ==  "true" ]; then
  if [ -e "/dev/tdx-guest" ]; then
    cd /opt
    bash /opt/attestation.sh
    bash /opt/temp_command_file
    if [ $? -ne 0 ]; then
      echo "[ERROR] Attestation Failed!"
      exit 1
    fi
  else
      echo "TDX device not found!"
  fi
fi

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
dispatch_method="shortest_queue" # shortest_queue or lottery

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
  exec /usr/bin/tini -s -- "bash"
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
        ;;
      -h|--help)
        usage
        ;;
      --)
        shift
        break
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

  if [[ -n $DISPATCH_METHOD ]]; then
    dispatch_method=$DISPATCH_METHOD
  fi

  if [[ $ENABLE_ATTESTATION_API = "true" ]]; then
    attest_flag="--attest"
    echo 'port=4050' | tee /etc/tdx-attest.conf
  fi

  controller_address="http://$controller_host:$controller_port"
  # Execute logic based on options
  if [[ $mode == "controller" ]]; then
    # Logic for controller mode
    # Boot Controller
    api_address="http://$api_host:$api_port"
    echo "Controller address: $controller_address"
    echo "OpenAI API address: $api_address"
    python3 -m fastchat.serve.controller --host $controller_host --port $controller_port --dispatch-method $dispatch_method $attest_flag &
    # Boot openai api server
    python3 -m fastchat.serve.openai_api_server --host $api_host --port $api_port --controller-address $controller_address $attest_flag
  else
    # Logic for non-controller(worker) mode
    worker_address="http://$worker_host:$worker_port"
    # Apply optimizations from bigdl-nano
    source bigdl-nano-init -t
    # First check if user have set OMP_NUM_THREADS by themselves
    if [[ -n "${omp_num_threads}" ]]; then
      echo "Setting OMP_NUM_THREADS to its original value: $omp_num_threads"
      export OMP_NUM_THREADS=$omp_num_threads
    else
      # Use calculate_total_cores to acquire cpuset settings
      # Set OMP_NUM_THREADS to correct numbers
      cores=$(calculate_total_cores)
      if [[ $cores == -1 || $cores == 0 ]]; then
        echo "Failed to obtain the number of cores, will use the default settings OMP_NUM_THREADS=$OMP_NUM_THREADS"
      else
        echo "Setting OMP_NUM_THREADS to $cores"
        export OMP_NUM_THREADS=$cores
      fi
    fi
    if [[ -z "${model_path}" ]]; then
          echo "Please set env MODEL_PATH used for worker"
          usage
    fi
    echo "Worker address: $worker_address"
    echo "Controller address: $controller_address"
    python3 -m fastchat.serve.model_worker --model-path $model_path --device cpu --host $worker_host --port $worker_port --worker-address $worker_address --controller-address $controller_address $attest_flag
  fi
fi

