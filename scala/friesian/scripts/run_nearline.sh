#!/bin/bash
set -e

docker_name=intelanalytics/friesian-serving:2.2.0-SNAPSHOT

docker run -it --net host --rm -v $(pwd):/opt/work/mnt $docker_name feature-init -c mnt/nearline/config_feature.yaml

docker run -it --net host --rm -v $(pwd):/opt/work/mnt $docker_name feature-init -c mnt/nearline/config_feature_vec.yaml

docker run -it --net host --rm -v $(pwd):/opt/work/mnt $docker_name recall-init -c mnt/nearline/config_recall.yaml

