#!/bin/bash
set -e

echo "Starting ranking services..."
bash deploy.sh ranking 8083 7083 config_ranking.yaml

echo "Starting feature services..."
bash deploy.sh feature 8082 7082 config_feature.yaml

echo "Starting feature recall services..."
bash deploy.sh feature_recall 8085 7085 config_feature_recall.yaml

echo "Starting recall services..."
bash deploy.sh recall 8084 7084 config_recall.yaml 

echo "Starting recommender services..."
bash deploy.sh recommender 8980 7980 config_recommender.yaml


