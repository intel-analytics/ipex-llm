#!/bin/bash
set -e

docker_name=intelanalytics/friesian-serving:2.2.0-SNAPSHOT

cd ..
onlinePath=src/main/resources/config

docker run -itd --net host  --rm --name ranking -v $(pwd):/opt/work/mnt -e OMP_NUM_THREADS=1 $docker_name ranking -c mnt/$onlinePath/config_ranking.yaml

docker run -itd --net host --rm --name feature -v $(pwd):/opt/work/mnt $docker_name feature -c mnt/$onlinePath/config_feature.yaml

docker run -itd --net host --rm --name feature_recall -v $(pwd):/opt/work/mnt $docker_name feature -c mnt/$onlinePath/config_feature_vec.yaml

docker run -itd --net host --rm --name recall -v $(pwd):/opt/work/mnt $docker_name recall -c mnt/$onlinePath/config_recall.yaml

docker run -itd --net host  --rm --name recommender_http -v $(pwd):/opt/work/mnt $docker_name recommender-http -c mnt/$onlinePath/config_recommender.yaml -p 8000
