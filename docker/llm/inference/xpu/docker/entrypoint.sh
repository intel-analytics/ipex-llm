#!/bin/bash

usage() {
  echo "Usage: $0 [-h --help] [--repo-id <repo_id>] [--test-api <test_api>] [--in-out-pairs <in_out_pairs>]"
  echo "-h: Print help message."
  echo "--repo-id: Specify the repo_id (comma-separated if multiple values)."
  echo "--test-api: Specify the test API (comma-separated if multiple values)."
  echo "--in-out-pairs: Specify the in_out_pairs values (comma-separated if multiple values)."
  exit 1
}

# We do not have any arguments, just run bash
if [ "$#" == 0 ]; then
  echo "[INFO] no command is passed in"
  echo "[INFO] enter pass-through mode"
  exec /usr/bin/tini -s -- "bash"
else
  # Parse command-line options
  options=$(getopt -o "m:hw:" --long "repo-id:,help,test-api:,in-out-pairs:" -n "$0" -- "$@")
  if [ $? != 0 ]; then
    usage
  fi
  eval set -- "$options"

  while true; do
    case "$1" in
      --repo-id)
        repo_ids="$2"
        shift 2
        ;;
      --test-api)
        test_apis="$2"
        # 检查 test_api 是否在可选列表中
        valid_test_apis=(
          "transformer_int4_gpu" 
          "transformer_int4_fp16_gpu"
          "ipex_fp16_gpu"
          "bigdl_fp16_gpu"
          "optimize_model_gpu" 
          "transformer_int4_gpu_win"
          "transformer_int4_fp16_gpu_win"
          "transformer_int4_loadlowbit_gpu_win"
          "deepspeed_optimize_model_gpu"
          "pipeline_parallel_gpu"
          "speculative_gpu"
          "transformer_int4"
          "native_int4"
          "optimize_model"
          "pytorch_autocast_bf16"
          "transformer_autocast_bf16"
          "bigdl_ipex_bf16"
          "bigdl_ipex_int4"
          "bigdl_ipex_int8"
          "speculative_cpu"
          "deepspeed_transformer_int4_cpu"
        )
        for api in $(echo "$test_apis" | tr ',' '\n'); do
          if ! [[ " ${valid_test_apis[@]} " =~ " ${api} " ]]; then
            echo "Invalid test API: $api"
            exit 1
          fi
        done
        shift 2
        ;;
      --in-out-pairs)
        in_out_pairs="$2"
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
fi

# Replace local_model_hub
sed -i "s/'path to your local model hub'/'\/llm\/models'/" config.yaml

# Comment out repo_id, in_out_pairs and test_api
sed -i -E "/^(\s*-)/s|^|  #|" config.yaml

# Modify config.yaml with repo_id
if [ -n "$repo_ids" ]; then
  for repo_id in $(echo "$repo_ids" | tr ',' '\n'); do
    # Add each repo_id value as a subitem of repo_id list
    sed -i -E "/^(repo_id:)/a \  - '$repo_id'" config.yaml
  done
fi

# Modify config.yaml with test_api
if [ -n "$test_apis" ]; then
  for test_api in $(echo "$test_apis" | tr ',' '\n'); do
    # Add each test_api value as a subitem of test_api list
    sed -i -E "/^(test_api:)/a \  - '$test_api'" config.yaml
  done
fi

# Modify config.yaml with in_out_pairs
if [ -n "$in_out_pairs" ]; then
  for in_out_pair in $(echo "$in_out_pairs" | tr ',' '\n'); do
    # Add the in_out_pairs values as subitems of in_out_pairs list
    sed -i -E "/^(in_out_pairs:)/a \  - '$in_out_pair'" config.yaml
  done
fi

exec /usr/bin/bash -s -- "bash"