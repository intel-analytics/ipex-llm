FROM ubuntu:22.04

ARG http_proxy
ARG https_proxy

RUN mkdir -p /ppml/


# Avoid build stuck when installing tzdata
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install PYTHON 3.9
RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.9 && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.9 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get install -y python3-pip python3.9-dev python3-wheel && \
    pip3 install --upgrade pip && \
    pip3 install --no-cache requests argparse cryptography==3.3.2 urllib3 && \
    pip3 install --upgrade requests && \
    pip3 install setuptools==58.4.0 && \
# Install bigdl-llm
    pip install --pre --upgrade bigdl-llm[all] && \
# Remove all dependencies from nvidia.
    pip3 list | grep nvidia | awk '{print $1}' | xargs pip3 uninstall -y



ENV PYTHONPATH   /usr/lib/python3.9:/usr/lib/python3.9/lib-dynload:/usr/local/lib/python3.9/dist-packages:/usr/lib/python3/dist-packages:/ppml/bigdl-ppml/src

WORKDIR /ppml
ENTRYPOINT [ "/bin/bash" ]
