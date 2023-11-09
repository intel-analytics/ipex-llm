#!/bin/bash

usage() {
  echo "Usage: $0 [--service-model-path <service model path>] [--help]"
  echo "--help: Print help message."
  echo "--service-model-path: set model path for model worker"
  echo "The following environment variables can be set."
  echo "CONTROLLER_HOST (default: localhost)."
  echo "CONTROLLER_PORT (default: 21001)."
  echo "WORKER_HOST (default: localhost)."
  echo "WORKER_PORT (default: 21002)."
  echo "API_HOST (default: localhost)."
  echo "API_PORT (default: 8000)."
  exit 1
}

# Default values
controller_host="localhost"
controller_port="21001"
worker_host="localhost"
worker_port="21002"
api_host="localhost"
api_port="8000"
service_model_path=""

# We do not have any arguments, just run bash
if [ "$#" == 0 ]; then
  echo "[INFO] no command is passed in"
  echo "[INFO] enter pass-through mode"
  exec /usr/bin/bash -s -- "bash"
else
  # Parse command-line options
  options=$(getopt -o "" --long "service-model-path:,help" -n "$0" -- "$@")
  if [ $? != 0 ]; then
    usage
  fi
  eval set -- "$options"

  while true; do
    case "$1" in
    --service-model-path)
        service_model_path="$2"
        shift 2
        ;;
    --help)
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

  if [[ -n $WORKER_HOST ]]; then
    worker_host=$WORKER_HOST
  fi

  if [[ -n $WORKER_PORT ]]; then
    worker_port=$WORKER_PORT
  fi

  if [[ -n $API_HOST ]]; then
    api_host=$API_HOST
  fi

  if [[ -n $API_PORT ]]; then
    api_port=$API_PORT
  fi

  controller_address="http://$controller_host:$controller_port"
  worker_address="http://$worker_host:$worker_port"
  api_address="http://$api_host:$api_port"

  unset http_proxy
  unset https_proxy

  python3 -m fastchat.serve.controller --host $controller_host --port $controller_port &
  python3 -m bigdl.llm.serving.model_worker --model-path $service_model_path --device xpu --host $worker_host --port $worker_port --worker-address $worker_address --controller-address $controller_address &
  python3 -m fastchat.serve.openai_api_server --host $api_host --port $api_port --controller-address $controller_address &

  echo "Controller address: $controller_address"
  echo "Worker address: $worker_address"
  echo "OpenAI API address: $api_address"

fi

exec /usr/bin/bash -s -- "bash"
