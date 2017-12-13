--------
# Dockfiles for spark-on-k8s

## Before the build

Please download the spark-2.2.0-k8s distribution like spark-2.2.0-k8s-0.5.0-bin-with-hadoop-2.7.3.tgz and unzip it. Replace the dockerfiles with dockerfiles here.

## How to build it

### First, build the latest images for spark-base, spark-driver, spark-driver-py, spark-executor, spark-executor-py and spark-init.

    sudo docker build -t spark-base:latest \
        -f dockerfiles/spark-base/Dockerfile . \
        --build-arg http_proxy=http://your-proxy-host:your-proxy-port \
        --build-arg https_proxy=https://your-proxy-host:your-proxy-port
    sudo docker build -t spark-driver:latest \ 
        -f dockerfiles/driver/Dockerfile . \
        --build-arg http_proxy=http://your-proxy-host:your-proxy-port \
        --build-arg https_proxy=https://your-proxy-host:your-proxy-port
    sudo docker build -t spark-driver-py:latest \
        -f dockerfiles/driver-py/Dockerfile . \
        --build-arg http_proxy=http://your-proxy-host:your-proxy-port \
        --build-arg https_proxy=https://your-proxy-host:your-proxy-port
    sudo docker build -t spark-executor:latest \
        -f dockerfiles/executor/Dockerfile . \
        --build-arg http_proxy=http://your-proxy-host:your-proxy-port \
        --build-arg https_proxy=https://your-proxy-host:your-proxy-port
    sudo docker build -t spark-executor-py:latest \
        -f dockerfiles/executor-py/Dockerfile . \
        --build-arg http_proxy=http://your-proxy-host:your-proxy-port \
        --build-arg https_proxy=https://your-proxy-host:your-proxy-port
    sudo docker build -t spark-init:latest \
        -f dockerfiles/init-container/Dockerfile . \
        --build-arg http_proxy=http://your-proxy-host:your-proxy-port \
        --build-arg https_proxy=https://your-proxy-host:your-proxy-port

### Then tag these images with a meaning tag.
    
    sudo docker tag spark-base:latest intelanalytics/spark-base:v2.2.0-kubernetes-0.5.0-ubuntu-14.04
    sudo docker tag spark-driver:latest intelanalytics/spark-driver:v2.2.0-kubernetes-0.5.0-ubuntu-14.04
    sudo docker tag spark-executor:latest intelanalytics/spark-executor:v2.2.0-kubernetes-0.5.0-ubuntu-14.04
    sudo docker tag spark-init:latest intelanalytics/spark-init:v2.2.0-kubernetes-0.5.0-ubuntu-14.04
    sudo docker tag spark-driver-py:latest intelanalytics/spark-driver-py:v2.2.0-kubernetes-0.5.0-ubuntu-14.04
    sudo docker tag spark-executor-py:latest intelanalytics/spark-executor-py:v2.2.0-kubernetes-0.5.0-ubuntu-14.04
