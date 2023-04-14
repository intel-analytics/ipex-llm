FROM intelanalytics/bigdl-k8s

ADD ./download-bigdl-ppml.sh /opt/download-bigdl-ppml.sh
RUN chmod a+x /opt/download-bigdl-ppml.sh

RUN apt-get update --fix-missing && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y tzdata && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
# Install python3.8
    apt-get install -y python3.8 python3.8-dev python3.8-distutils build-essential python3-wheel python3-pip && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.8 /usr/bin/python3 && \
    pip3 install --upgrade pip && \
    pip3 install setuptools && \
    pip3 install numpy && \
    ln -s /usr/bin/python3 /usr/bin/python && \
# Download BigDL PPML jar with dependency jar
    /opt/download-bigdl-ppml.sh

ENV PYTHONPATH  /usr/lib/python3.8:/usr/lib/python3.8/lib-dynload:/usr/local/lib/python3.8/dist-packages:/usr/lib/python3/dist-packages
