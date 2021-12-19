--------
# Docker images and builders for BigDL

## BigDL in Docker

### By default, the BigDL image has installed below packages:
* git
* maven
* Oracle jdk 1.8.0_152 (in /opt/jdk1.8.0_152)
* python 2.7.6
* pip
* numpy
* scipy
* pandas
* scikit-learn
* matplotlib
* seaborn
* jupyter
* wordcloud
* spark-${SPARK_VERSION} (in /opt/work/spark-${SPARK_VERSION})
* bigdl distribution (in /opt/work/bigdl-${BIGDL_VERSION})
* BigDL-Tutorials (in /opt/work/BigDL-Tutorials)

### The work dir for BigDL is /opt/work.

* download-bigdl.sh is used for downloading BigDL distributions.
* start-notebook.sh is used for starting the jupyter notebook. You can specify the environment settings and spark settings to start a specified jupyter notebook.
* bigdl-${BIGDL_VERSION} is the BigDL home of BigDL distribution.
* dist-spark-x.x.x-scala-x.x.x-all-x.x.x-dist.zip is the zip file of BigDL distribution.
* spark-${SPARK_VERSION} is the Spark home.
* BigDL-Tutorials is cloned from https://github.com/intel-analytics/BigDL-Tutorials, contains the Deep Leaning Tutorials on Apache Spark using BigDL.

## How to build it.

### By default, you can build a bigdl:default image with latest nightly-build BigDL distributions:

    sudo docker build --rm -t bigdl/bigdl:default .

### If you need http and https proxy to build the image:

    sudo docker build \
        --build-arg http_proxy=http://your-proxy-host:your-proxy-port \
        --build-arg https_proxy=https://your-proxy-host:your-proxy-port \
        --rm -t bigdl/bigdl:default .

### You can also specify the BIGDL_VERSION and SPARK_VERSION to build a specific BigDL image:

    sudo docker build \
        --build-arg http_proxy=http://your-proxy-host:your-proxy-port \
        --build-arg https_proxy=https://your-proxy-host:your-proxy-port \
        --build-arg BIGDL_VERSION=0.2.0 \
        --build-arg SPARK_VERSION=2.1.1 \
        --rm -t bigdl/bigdl:0.2.0-spark-2.1.1 .

## How to use the image.

### To start a notebook directly with a specified port(e.g. 12345). You can view the notebook on http://[host-ip]:12345

    sudo docker run -it --rm -p 12345:12345 -e NotebookPort=12345 -e NotebookToken=your-token bigdl/bigdl:default

    sudo docker run -it --rm --net=host -e NotebookPort=12345 -e NotebookToken=your-token bigdl/bigdl:default

    sudo docker run -it --rm -p 12345:12345 -e NotebookPort=12345 -e NotebookToken=your-token bigdl/bigdl:0.2.0-spark-2.1.1

    sudo docker run -it --rm --net=host -e NotebookPort=12345 -e NotebookToken=your-token bigdl/bigdl:0.2.0-spark-2.1.1

### If you need http and https proxy in your environment:

    sudo docker run -it --rm -p 12345:12345 -e NotebookPort=12345 -e NotebookToken=your-token -e http_proxy=http://your-proxy-host:your-proxy-port -e https_proxy=https://your-proxy-host:your-proxy-port bigdl/bigdl:default

    sudo docker run -it --rm --net=host -e NotebookPort=12345 -e NotebookToken=your-token -e http_proxy=http://your-proxy-host:your-proxy-port -e https_proxy=https://your-proxy-host:your-proxy-port  bigdl/bigdl:default

    sudo docker run -it --rm -p 12345:12345 -e NotebookPort=12345 -e NotebookToken=your-token -e http_proxy=http://your-proxy-host:your-proxy-port -e https_proxy=https://your-proxy-host:your-proxy-port  bigdl/bigdl:0.2.0-spark-2.1.1

    sudo docker run -it --rm --net=host -e NotebookPort=12345 -e NotebookToken=your-token -e http_proxy=http://your-proxy-host:your-proxy-port -e https_proxy=https://your-proxy-host:your-proxy-port bigdl/bigdl:0.2.0-spark-2.1.1

### You can also start the container first

    sudo docker run -it --rm --net=host -e NotebookPort=12345 -e NotebookToken=your-token bigdl/bigdl:default bash

### In the container, after setting proxy and ports, you can start the Notebook by:

    /opt/work/start-notebook.sh
