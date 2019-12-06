docker build \
    --build-arg http_proxy=http://10.239.4.101:913/ \
    --build-arg https_proxy=https://10.239.4.101:913/ \
    --build-arg SPARK_VERSION=2.4.0 \
    --rm -t analytics-zoo/cluster-serving:0.7.0-spark_2.4.0 \
    -f ./docker/DockerFile .
