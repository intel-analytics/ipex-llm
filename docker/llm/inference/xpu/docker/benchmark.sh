#!/bin/bash

echo "Repo ID is: $REPO_IDS"
echo "Test API is: $TEST_APIS"
echo "In-out pairs are: $IN_OUT_PAIRS"
echo "Device is: $DEVICE"

cd /benchmark/all-in-one

# Replace local_model_hub
sed -i "s/'path to your local model hub'/'\/llm\/models'/" config.yaml

# Comment out repo_id, in_out_pairs and test_api
sed -i -E "/^(\s*-)/s|^|  #|" config.yaml

# Modify config.yaml with repo_id
if [ -n "$REPO_IDS" ]; then
  for REPO_ID in $(echo "$REPO_IDS" | tr ',' '\n'); do
    # Add each repo_id value as a subitem of repo_id list
    sed -i -E "/^(repo_id:)/a \  - '$REPO_ID'" config.yaml
  done
fi

# Modify config.yaml with test_api
if [ -n "$TEST_APIS" ]; then
  for TEST_API in $(echo "$TEST_APIS" | tr ',' '\n'); do
    # Add each test_api value as a subitem of test_api list
    sed -i -E "/^(test_api:)/a \  - '$TEST_API'" config.yaml
  done
fi

# Modify config.yaml with in_out_pairs
if [ -n "$IN_OUT_PAIRS" ]; then
  for IN_OUT_PAIRS in $(echo "$IN_OUT_PAIRS" | tr ',' '\n'); do
    # Add the in_out_pairs values as subitems of in_out_pairs list
    sed -i -E "/^(in_out_pairs:)/a \  - '$IN_OUT_PAIR'" config.yaml
  done
fi


if [[ "$DEVICE" == "Arc" || "$DEVICE" == "ARC" ]]; then
    source ipex-llm-init -g --device Arc
    python run.py
elif [[ "$DEVICE" == "Flex" || "$DEVICE" == "FLEX" ]]; then
    source ipex-llm-init -g --device Flex
    python run.py
elif [[ "$DEVICE" == "Max" || "$DEVICE" == "MAX" ]]; then
    source ipex-llm-init -g --device Max
    python run.py
else
    echo "Invalid DEVICE specified."
fi
