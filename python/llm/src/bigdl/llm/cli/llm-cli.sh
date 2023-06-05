#!/bin/bash

# Default values
model_family=""

llm_dir="$(dirname "$(python -c "import bigdl.llm;print(bigdl.llm.__file__)")")"
lib_dir="$llm_dir/libs"

# Function to display help message
function display_help {
  echo "usage: ./llm-cli.sh -x MODEL_FAMILY [-h] [args]"
  echo ""
  echo "options:"
  echo "  -h, --help  show this help message"
  echo "  -x, --model_family {llama,bloom,gptneox}"
  echo "              family name of model"
  echo "  args        parameters passed to the specified model function"
}

function llama {
  command="$lib_dir/main-llama $filteredArguments"
  $command
}

function bloom {
  command="$lib_dir/main-bloom $filteredArguments"
  $command
}

function gptneox {
  command="$lib_dir/main-gptneox $filteredArguments"
  $command
}
# Remove model_family/x parameter
filteredArguments=()
while [[ $# -gt 0 ]]; do
  case "$1" in
  -h | --help)
    display_help
    filteredArguments+=("$1")
    shift
    ;;
  -x | --model_family)
    model_family="$2"
    shift 2
    ;;
  *)
    filteredArguments+=("$1")
    shift
    ;;
  esac
done

# Perform actions based on the model_family
if [[ "$model_family" == "llama" ]]; then
  llama
elif [[ "$model_family" == "bloom" ]]; then
  bloom
elif [[ "$model_family" == "gptneox" ]]; then
  gptneox
else
  echo "Invalid model_family: $model_family"
  display_help
fi
