#!/bin/bash

usage() {
  echo "Usage: $0 [-c] [-h] [-z <controller_host>] [-p <controller_port>] [-o <api_host>] [-u <api_port>] [-x <worker_host>] [-y <worker_port>] [-m <model_path>]"
  echo "-c: Use controller mode."
  echo "-h: Print help message."
  echo "-z: Set the controller host (default: localhost)."
  echo "-p: Set the controller port (default: 21001)."
  echo "-o: Set the API host (default: localhost)."
  echo "-u: Set the API port (default: 8000)."
  echo "-x: Set the worker host (default: localhost). (Only applicable in non-controller mode.)"
  echo "-y: Set the worker port (default: 21002). (Only applicable in non-controller mode.)"
  echo "-m: Set the worker model path(default: empty). (Only applicable in non-controller mode.)"
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
  else
    echo -1
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

# We do not have any arguments, just run bash
if [ "$#" == 0 ]; then
  echo "[INFO] no command is passed in"
  echo "[INFO] enter pass-through mode"
  exec /usr/bin/tini -s -- "bash"
else
  # Parse command-line options
  while getopts "chz:p:o:u:x:y:m:" opt; do
    case ${opt} in
      c)
        controller_mode=true
        ;;
      h)
        usage
        ;;
      z)
        controller_host=${OPTARG}
        ;;
      p)
        controller_port=${OPTARG}
        ;;
      o)
        api_host=${OPTARG}
        ;;
      u)
        api_port=${OPTARG}
        ;;
      x)
        if [[ -n $controller_mode ]]; then
          echo "Invalid option: -wh cannot be used with -c."
          usage
        fi
        worker_host=${OPTARG}
        ;;
      y)
        if [[ -n $controller_mode ]]; then
          echo "Invalid option: -wp cannot be used with -c."
          usage
        fi
        worker_port=${OPTARG}
        ;;
      m)
        if [[ -n $controller_mode ]]; then
          echo "Invalid option: -mp cannot be used with -c."
          usage
        fi
        model_path=${OPTARG}
        ;;

      *)
        usage
        ;;
    esac
  done

  shift $((OPTIND - 1))

  controller_address="http://$controller_host:$controller_port"
  # Execute logic based on options
  if [[ -n $controller_mode ]]; then
    # Logic for controller mode
    # Boot Controller
    # TODO: add dispatch-method
    api_address="http://$api_host:$api_port"
    echo "Controller address: $controller_address"
    echo "OpenAI API address: $api_address"
    python3 -m fastchat.serve.controller --host $controller_host --port $controller_port &
    # Boot openai api server
    python3 -m fastchat.serve.openai_api_server --host $api_host --port $api_port --controller-address $controller_address
  else
    # Logic for non-controller(worker) mode
    worker_address="http://$worker_host:$worker_port"
    # Apply optimizations from bigdl-nano
    source bigdl-nano-init -t
    # Set OMP_NUM_THREADS to correct numbers
    # This works in TDX-CoCo, TDX-VM, and native
    cores=$(calculate_total_cores)
    if [[ $cores == -1 ]]; then
      echo "Failed to obtain the number of cores, will use the default settings OMP_NUM_THREADS=$OMP_NUM_THREADS"
    else
      echo "Setting OMP_NUM_THREADS to $cores"
      export OMP_NUM_THREADS=$cores
    fi
    if [[ -z "${model_path}" ]]; then
          echo "Please set model path used for worker"
          usage
    fi
    echo "Worker address: $worker_address"
    echo "Controller address: $controller_address"
    python3 -m fastchat.serve.model_worker --model-path $model_path --device cpu --host $worker_host --port $worker_port --worker-address $worker_address
  fi
fi

