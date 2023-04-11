FROM intelanalytics/bigdl-k8s

# Install python3.8
RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.8 python3.8-dev python3.8-distutils build-essential python3-wheel python3-pip && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.8 /usr/bin/python3 && \
    pip3 install --upgrade pip && \
    pip3 install setuptools && \
    ln -s /usr/bin/python3 /usr/bin/python

ENV PYTHONPATH  /usr/lib/python3.8:/usr/lib/python3.8/lib-dynload:/usr/local/lib/python3.8/dist-packages:/usr/lib/python3/dist-packages


# Download BigDL PPML jar with dependency jar
ADD ./download-bigdl-ppml.sh /opt/download-bigdl-ppml.sh
RUN chmod a+x /opt/download-bigdl-ppml.sh
RUN /opt/download-bigdl-ppml.sh

ADD ./entrypoint.sh /opt/entrypoint.sh
ENTRYPOINT [ "/opt/entrypoint.sh" ]
