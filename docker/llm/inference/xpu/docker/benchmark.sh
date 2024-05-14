#!/bin/bash

echo "Repo ID is: $REPO_IDS"
echo "Test API is: $TEST_APIS"
echo "Device is: $DEVICE"

cd /benchmark/all-in-one

# Replace local_model_hub
sed -i "s/'path to your local model hub'/'\/llm\/models'/" config.yaml

# Comment out repo_id
sed -i -E "/repo_id:/,/local_model_hub/ s/^(\s*-)/  #&/" config.yaml

# Modify config.yaml with repo_id
if [ -n "$REPO_IDS" ]; then
  for REPO_ID in $(echo "$REPO_IDS" | tr ',' '\n'); do
    # Add each repo_id value as a subitem of repo_id list
    sed -i -E "/^(repo_id:)/a \  - '$REPO_ID'" config.yaml
  done
fi

# Comment out test_api
sed -i -E "/test_api:/,/cpu_embedding/ s/^(\s*-)/  #&/" config.yaml

# Modify config.yaml with test_api
if [ -n "$TEST_APIS" ]; then
  for TEST_API in $(echo "$TEST_APIS" | tr ',' '\n'); do
    # Add each test_api value as a subitem of test_api list
    sed -i -E "/^(test_api:)/a \  - '$TEST_API'" config.yaml
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

# print out results
for file in *.csv; do
    echo ""
    echo "filename: $file"
    cat "$file"
done
