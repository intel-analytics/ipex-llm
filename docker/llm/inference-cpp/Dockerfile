FROM intel/oneapi-basekit:2025.0.2-0-devel-ubuntu24.04

ARG http_proxy
ARG https_proxy

ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1

# When cache is enabled SYCL runtime will try to cache and reuse JIT-compiled binaries. 
ENV SYCL_CACHE_PERSISTENT=1

# Disable pip's cache behavior
ARG PIP_NO_CACHE_DIR=false

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/intel-oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main " | tee /etc/apt/sources.list.d/oneAPI.list && \
    chmod 644 /usr/share/keyrings/intel-oneapi-archive-keyring.gpg && \
    rm /etc/apt/sources.list.d/intel-graphics.list && \
    wget -O- https://repositories.intel.com/graphics/intel-graphics.key | gpg --dearmor | tee /usr/share/keyrings/intel-graphics.gpg > /dev/null && \
    echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy arc" | tee /etc/apt/sources.list.d/intel.gpu.jammy.list && \
    chmod 644 /usr/share/keyrings/intel-graphics.gpg && \
    apt-get update && \
    apt-get install -y --no-install-recommends curl wget git sudo && \
    # Install PYTHON 3.11 and IPEX-LLM[xpu]
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    env DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends libunwind8-dev vim less && \
    # add-apt-repository requires gnupg, gpg-agent, software-properties-common
    apt-get install -y --no-install-recommends gnupg gpg-agent software-properties-common && \
    # Add Python 3.11 PPA repository
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y --no-install-recommends python3.11 git curl wget && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.11 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get install -y --no-install-recommends python3.11-dev python3-wheel python3.11-distutils && \
    wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py && \
    pip install --upgrade requests argparse urllib3 && \
    pip install --pre --upgrade ipex-llm[cpp] && \
    # Fix Trivy CVE Issues
    pip install transformers==4.36.2 && \
    pip install transformers_stream_generator einops tiktoken && \
    # Install opencl-related repos
    apt-get update && \
    # apt-get install -y --no-install-recommends intel-opencl-icd intel-level-zero-gpu=1.3.26241.33-647~22.04 level-zero level-zero-dev --allow-downgrades && \
    # Remove breaks install packages
    apt-get remove -y libze-dev libze-intel-gpu1 && \
    mkdir -p /tmp/gpu && \
    cd /tmp/gpu && \
    # wget https://github.com/oneapi-src/level-zero/releases/download/v1.18.5/level-zero_1.18.5+u22.04_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-core-2_2.5.6+18417_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-opencl-2_2.5.6+18417_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu-dbgsym_1.6.32224.5_amd64.ddeb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu_1.6.32224.5_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd-dbgsym_24.52.32224.5_amd64.ddeb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd_24.52.32224.5_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/libigdgmm12_22.5.5_amd64.deb && \
    dpkg -i *.deb && \
    rm *.deb && \
    rm -rf /tmp/gpu && \
    mkdir -p /llm && \
    cd /llm && \
    # Install Dependencies
    # remove blinker to avoid error
    find /usr/lib/python3/dist-packages/ -name 'blinker*' -exec rm -rf {} + && \
    rm -rf /root/.cache/Cypress && \
    pip uninstall -y gunicorn python-jose PyMySQL

COPY ./start-llama-cpp.sh /llm/scripts/start-llama-cpp.sh
COPY ./start-ollama.sh /llm/scripts/start-ollama.sh
COPY ./benchmark_llama-cpp.sh /llm/scripts/benchmark_llama-cpp.sh

WORKDIR /llm/
