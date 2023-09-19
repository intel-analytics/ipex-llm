FROM intel/oneapi-basekit

ENV http_proxy $HTTP_PROXY
ENV https_proxy $HTTP_PROXY

ENV TZ=Asia/Shanghai

# Otherwise will require user to input timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y curl wget git gnupg gpg-agent

RUN wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
    echo 'deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy arc' | tee /etc/apt/sources.list.d/intel.gpu.jammy.list && \
    rm /etc/apt/sources.list.d/intel-graphics.list

# Install PYTHON 3.9 and BigDL-LLM[xpu]
RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt install software-properties-common libunwind8-dev vim less -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.9 git curl wget && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.9 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get install -y python3-pip python3.9-dev python3-wheel python3.9-distutils && \
    pip3 install --no-cache --upgrade requests argparse urllib3 && \
    pip3 install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu && \
    # Install opencl-related repos
    apt-get update && \
    apt-get install -y intel-opencl-icd intel-level-zero-gpu level-zero level-zero-dev
