--------
# Docker images and builders for BigDL

## BigDL in Docker

### By default, the BigDL image has installed below packages:
* git
* maven
* Oracle jdk 1.8.0_152 (in /opt/jdk1.8.0_152)
* python 3.6.9
* pip
* numpy
* scipy
* pandas
* scikit-learn
* matplotlib
* seaborn
* jupyter
* wordcloud
* moviepy
* requests
* tensorflow
* spark-${SPARK_VERSION} (in /opt/work/spark-${SPARK_VERSION})
* BigDL distribution (in /opt/work/BigDL-${BigDL_VERSION})
* BigDL source code (in /opt/work/BigDL)

### The work dir for BigDL is /opt/work.

* downlown-bigdl.sh is used for downloading BigDL distributions.
* BigDL-${BigDL_VERSION} is the BigDL home of BigDL distribution.
* BigDL-SPARK_x.x-x.x.x-dist.zip is the zip file of BigDL distribution.
* spark-${SPARK_VERSION} is the Spark home.
* BigDL is cloned from https://github.com/intel-analytics/BigDL, contains apps, examples using BigDL.

## How to build it.

### By default, you can build a BigDL:default image with latest nightly-build BigDL distributions, you also need to provide jdk version and jdk download url:

    sudo docker build \
        --build-arg JDK_VERSION=8u192 \
        --build-arg JDK_URL=http://your-http-url-to-download-jdk \
        --rm -t intelanalytics/bigdl:default .

### If you need http and https proxy to build the image:

    sudo docker build \
        --build-arg http_proxy=http://your-proxy-host:your-proxy-port \
        --build-arg https_proxy=https://your-proxy-host:your-proxy-port \
        --build-arg JDK_VERSION=8u192 \
        --build-arg JDK_URL=http://your-http-url-to-download-jdk \
        --build-arg no_proxy=x.x.x.x \
        --rm -t intelanalytics/bigdl:default .

### You can also specify the BigDL_VERSION and SPARK_VERSION to build a specific BigDL image:

    sudo docker build \
        --build-arg http_proxy=http://your-proxy-host:your-proxy-port \
        --build-arg https_proxy=https://your-proxy-host:your-proxy-port \
        --build-arg JDK_VERSION=8u192 \
        --build-arg JDK_URL=http://your-http-url-to-download-jdk \
        --build-arg no_proxy=x.x.x.x \
        --build-arg BIGDL_VERSION=0.14.0 \
        --build-arg SPARK_VERSION=2.4.6 \
        --rm -t intelanalytics/bigdl:latest .

## How to use the image.

### To start a notebook directly with a specified port(e.g. 12345). You can view the notebook on http://[host-ip]:12345

    sudo docker run -it --rm -p 12345:12345 \
        -e NOTEBOOK_PORT=12345 \
        -e NOTEBOOK_TOKEN="your-token" \
        intelanalytics/bigdl:default

    sudo docker run -it --rm --net=host \
        -e NOTEBOOK_PORT=12345 \
        -e NOTEBOOK_TOKEN="your-token" \
        intelanalytics/bigdl:default

    sudo docker run -it --rm -p 12345:12345 \
        -e NOTEBOOK_PORT=12345 \
        -e NOTEBOOK_TOKEN="your-token" \
        intelanalytics/bigdl:latest

    sudo docker run -it --rm --net=host \
        -e NOTEBOOK_PORT=12345 \
        -e NOTEBOOK_TOKEN="your-token" \
        intelanalytics/bigdl:spark_2.4.6

### If you need http and https proxy in your environment:

    sudo docker run -it --rm -p 12345:12345 \
        -e NOTEBOOK_PORT=12345 \
        -e NOTEBOOK_TOKEN="your-token" \
        -e http_proxy=http://your-proxy-host:your-proxy-port \
        -e https_proxy=https://your-proxy-host:your-proxy-port \
        intelanalytics/bigdl:default

    sudo docker run -it --rm --net=host \
        -e NOTEBOOK_PORT=12345 \
        -e NOTEBOOK_TOKEN="your-token" \
        -e http_proxy=http://your-proxy-host:your-proxy-port \
        -e https_proxy=https://your-proxy-host:your-proxy-port \
        intelanalytics/bigdl:default

    sudo docker run -it --rm -p 12345:12345 \
        -e NOTEBOOK_PORT=12345 \
        -e NOTEBOOK_TOKEN="your-token" \
        -e http_proxy=http://your-proxy-host:your-proxy-port \
        -e https_proxy=https://your-proxy-host:your-proxy-port \
        intelanalytics/bigdl:spark_2.4.6

    sudo docker run -it --rm --net=host \
        -e NOTEBOOK_PORT=12345 \
        -e NOTEBOOK_TOKEN="your-token" \
        -e http_proxy=http://your-proxy-host:your-proxy-port \
        -e https_proxy=https://your-proxy-host:your-proxy-port \
        intelanalytics/bigdl:spark_2.4.6

### You can also start the container first

    sudo docker run -it --rm --net=host \
        -e NOTEBOOK_PORT=12345 \
        -e NOTEBOOK_TOKEN="your-token" \
        intelanalytics/bigdl:default bash

### In the container, after setting proxy and ports, you can start the Notebook by:

    /opt/work/start-notebook.sh

## Notice

### If you need nightly build version of BigDL, please pull the image form Dockerhub with:

    sudo docker pull intelanalytics/bigdl:latest

### Please follow the readme in each app folder to test the jupyter notebooks !!!

### With 0.3+ version of BigDL Docker image, you can specify the runtime conf of spark

    sudo docker run -itd --net=host \
        -e NOTEBOOK_PORT=12345 \
        -e NOTEBOOK_TOKEN="1234qwer" \
        -e http_proxy=http://your-proxy-host:your-proxy-port  \
        -e https_proxy=https://your-proxy-host:your-proxy-port  \
        -e RUNTIME_SPARK_MASTER=spark://your-spark-master-host:your-spark-master-port or local[*] \
        -e RUNTIME_DRIVER_CORES=4 \
        -e RUNTIME_DRIVER_MEMORY=20g \
        -e RUNTIME_EXECUTOR_CORES=4 \
        -e RUNTIME_EXECUTOR_MEMORY=20g \
        -e RUNTIME_TOTAL_EXECUTOR_CORES=4 \
        intelanalytics/bigdl:latest
