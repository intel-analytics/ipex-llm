#!/bin/bash

usage() {
  echo "Usage: $0 [-m --mode <controller|worker>] [-h --help] [-w --worker <model_worker|vllm_worker>]"
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
  echo "STREAM_INTERVAL (default: 1)."
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

# Default values
controller_host="localhost"
controller_port="21001"
gradio_port="8002"
api_host="localhost"
api_port="8000"
worker_host="localhost"
worker_port="21002"
model_path=""
mode=""
omp_num_threads=""
dispatch_method="shortest_queue" # shortest_queue or lottery
stream_interval=1
worker_type="model_worker"

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
  options=$(getopt -o "m:hw:" --long "mode:,help,worker:" -n "$0" -- "$@")
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
      -w|--worker)
        worker_type="$2"
        [[ $worker_type == "model_worker" || $worker_type == "vllm_worker" ]] || usage
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

  if [ "$worker_type" == "model_worker" ]; then
      worker_type="ipex_llm.serving.model_worker"
  elif [ "$worker_type" == "vllm_worker" ]; then
      worker_type="ipex_llm.serving.vllm_worker"
  fi

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

  if [[ -n $GRADIO_PORT ]]; then
    gradio_port=$GRADIO_PORT
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

  if [[ -n $STREAM_INTERVAL ]]; then
    stream_interval=$STREAM_INTERVAL
  fi

  controller_address="http://$controller_host:$controller_port"
  # Execute logic based on options
  if [[ $mode == "controller" ]]; then
    # Logic for controller mode
    # Boot Controller
    api_address="http://$api_host:$api_port"
    echo "Controller address: $controller_address"
    echo "OpenAI API address: $api_address"
    python3 -m fastchat.serve.controller --host $controller_host --port $controller_port --dispatch-method $dispatch_method &
    # Boot openai api server
    python3 -m fastchat.serve.openai_api_server --host $api_host --port $api_port --controller-address $controller_address &
    # Boot gradio_web_server
    python3 -m fastchat.serve.gradio_web_server --host $controller_host --port $gradio_port --controller-url $controller_address --model-list-mode reload
  else
    # Logic for non-controller(worker) mode
    worker_address="http://$worker_host:$worker_port"
    # Apply optimizations from ipex-llm
    source ipex-llm-init -t
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
    echo "Worker type: $worker_type"
    echo "Worker address: $worker_address"
    echo "Controller address: $controller_address"
    if [ "$worker_type" == "ipex_llm.serving.model_worker" ]; then
      python3 -m "$worker_type" --model-path $model_path --device cpu --host $worker_host --port $worker_port --worker-address $worker_address --controller-address $controller_address --stream-interval $stream_interval
    elif [ "$worker_type" == "ipex_llm.serving.vllm_worker" ]; then
      python3 -m "$worker_type" --model-path $model_path --device cpu --host $worker_host --port $worker_port --worker-address $worker_address --controller-address $controller_address
    fi
  fi
fi

exec /usr/bin/bash -s -- "bash"

