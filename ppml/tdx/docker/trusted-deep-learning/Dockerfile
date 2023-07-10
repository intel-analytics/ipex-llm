FROM ubuntu:20.04

RUN mkdir -p /ppml/

# Add example
RUN mkdir /ppml/examples
ADD ./pert.py                    /ppml/examples/pert.py
ADD ./pert_nano.py               /ppml/examples/pert_nano.py     

# Install PYTHON3.8
RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.8 python3.8-dev python3.8-distutils build-essential python3-wheel python3-pip && \
    apt-get install -y google-perftools=2.7-1ubuntu2 && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.8 /usr/bin/python3 && \
    pip3 install --upgrade pip && \
    pip3 install setuptools && \
    pip3 install --no-cache datasets==2.6.1 transformers intel_extension_for_pytorch && \
    ln -s /usr/bin/python3 /usr/bin/python

ENV PYTHONPATH  /usr/lib/python3.8:/usr/lib/python3.8/lib-dynload:/usr/local/lib/python3.8/dist-packages:/usr/lib/python3/dist-packages

RUN pip3 install --pre --no-cache --upgrade bigdl-nano[pytorch]
WORKDIR /ppml

ENTRYPOINT [ "/bin/bash" ]
