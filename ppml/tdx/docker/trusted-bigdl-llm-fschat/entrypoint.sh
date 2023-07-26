#!/bin/bash

usage() {
  echo "Usage: $0 [-c] [-ch <controller_host>] [-cp <controller_port>] [-oh <api_host>] [-op <api_port>] [-wh <worker_host>] [-wp <worker_port>]"
  echo "-c: Use controller mode."
  echo "-ch: Set the controller host (default: localhost)."
  echo "-cp: Set the controller port (default: 21001)."
  echo "-oh: Set the API host (default: localhost)."
  echo "-op: Set the API port (default: 8000)."
  echo "-wh: Set the worker host (default: localhost). (Only applicable in non-controller mode.)"
  echo "-wp: Set the worker port (default: 21002). (Only applicable in non-controller mode.)"
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

    echo "Total number of cores: $total_cores"
    return $total_cores
  else
    echo "cpuset.cpus file does not exist."
    return -1
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

# We do not have any arguments, just run bash
if [ "$#" == 0 ]; then
  echo "[INFO] no command is passed in"
  echo "[INFO] enter pass-through mode"
  exec /usr/bin/tini -s -- "bash"
else
  # Parse command-line options
  while getopts ":cch:cp:oh:op:wh:wp:" opt; do
    case ${opt} in
      c)
        controller_mode=true
        ;;
      ch)
        controller_host=${OPTARG}
        ;;
      cp)
        controller_port=${OPTARG}
        ;;
      oh)
        api_host=${OPTARG}
        ;;
      op)
        api_port=${OPTARG}
        ;;
      wh)
        if [[ -n $controller_mode ]]; then
          echo "Invalid option: -wh cannot be used with -c."
          usage
        fi
        worker_host=${OPTARG}
        ;;
      wp)
        if [[ -n $controller_mode ]]; then
          echo "Invalid option: -wp cannot be used with -c."
          usage
        fi
        worker_port=${OPTARG}
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
    # Logic for non-controller mode
    worker_address="http://$worker_host:$worker_port"
    echo "Worker address: $worker_address"
    echo "Controller address: $controller_address"

    # Apply optimizations from bigdl-nano
    source bigdl-nano-init -t
    # Set OMP_NUM_THREADS to correct numbers
    cores=$(calculate_total_cores)
    if [[ $cores == -1 ]]; then
      echo "Failed to obtain the number of cores, will use the default settings OMP_NUM_THREADS=$OMP_NUM_THREADS"
    else
      echo "Setting OMP_NUM_THREADS to $cores"
      export OMP_NUM_THREADS=$cores
    fi
  fi
fi

