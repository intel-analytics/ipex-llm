#!/bin/bash
set -e

printf "Shutdown ranking services....\n"
bash rm_cluster.sh ranking

printf "\nShutdown feature services....\n"
bash rm_cluster.sh feature

printf "\nShutdown feature recall services....\n"
bash rm_cluster.sh feature_recall

printf "\nShutdown recall services....\n"
bash rm_cluster.sh recall

printf "\nShutdown recommender services....\n"
bash rm_cluster.sh recommender
