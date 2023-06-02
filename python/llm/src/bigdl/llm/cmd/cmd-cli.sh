#!/bin/bash

# Default values
threads=28
n_predict=-1
interactive=false
model_family=""
prompt=empty
ctx_size=512

script_dir="$(dirname "$(readlink -f "$0")")"
lib_dir="$(dirname "$script_dir")/libs"

# Function to display help message
function display_help {
  echo "Usage: ./script.sh [OPTIONS]"
  echo "Options:"
  echo "  -c, --ctx_size N      size of the prompt context (default: 512)"
  echo "  -i, --interactive     run in interactive mode"
  echo "  -n, --n_predict N     number of tokens to predict (default: -1, -1 = infinity)"
  echo "  -m, --model FNAME     model path"
  echo "  -p, --prompt PROMPT   prompt to start generation with (default: empty)"
  echo "  -t, --threads N       number of threads to use during computation (default: 28)"
  echo "  -x, --model_family {llama,bloomz,gptneox}"
  echo "                        family name of model"
  echo "  -h, --help            display this help message"
  exit 1
}

function llama {
  command="$lib_dir/main-llama \
    -c $ctx_size \
    -n $n_predict \
    -m $model \
    -p $prompt \
    -t $threads"
  if [[ $interactive ]]; then
    command="$command -i"
  fi
  eval command
}
function bloomz {
  command="$lib_dir/main-bloomz \
    -c $ctx_size \
    -n $n_predict \
    -m $model \
    -p $prompt \
    -t $threads"
  if [[ $interactive ]]; then
    echo "Bloomz model family not support interactive mode"
    exit 1
  fi
  eval command
}
function gptneox {
  command="$lib_dir/main-gptneox \
    -c $ctx_size \
    -n $n_predict \
    -m $model \
    -p $prompt \
    -t $threads"
  if [[ $interactive ]]; then
    command="$command -i"
  fi
  eval command
}

# Parse command line options
while [[ $# -gt 0 ]]; do
  case "$1" in
  -c | --ctx_size)
    ctx_size="$2"
    shift 2
    ;;
  -t | --threads)
    threads="$2"
    shift 2
    ;;
  -p | --prompt)
    prompt="$2"
    shift 2
    ;;
  -n | --n_predict)
    n_predict="$2"
    shift 2
    ;;
  -m | --model)
    model="$2"
    shift 2
    ;;
  -i | --interactive)
    interactive=true
    shift 1
    ;;
  -x | --model_family)
    model_family="$2"
    shift 2
    ;;
  -h | --help)
    display_help
    ;;
  *)
    echo "Invalid option: $1"
    display_help
    ;;
  esac
done

# Print the values of the parsed arguments
echo "Threads: $threads"
echo "Prompt: $prompt"
echo "N Predict: $n_predict"
echo "Model: $model"
echo "Interactive Mode: $interactive"
echo "Model Family: $model_family"

# Perform actions based on the model_family
if [[ "$model_family" == "llama" ]]; then
  llama
elif [[ "$model_family" == "bloomz" ]]; then
  bloomz
elif [[ "$model_family" == "gptneox" ]]; then
  gptneox
else
  echo "Invalid model_family: $model_family"
  display_help
fi
