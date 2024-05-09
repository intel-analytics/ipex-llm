echo "Repo ID is: $REPO_ID"
echo "Test API is: $TEST_API"
echo "In-out pairs are: $IN_OUT_PAIRS"

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

