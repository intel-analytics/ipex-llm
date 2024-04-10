FROM ubuntu:22.04

ARG http_proxy
ARG https_proxy
ARG PIP_NO_CACHE_DIR=false
ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1

COPY ./start-notebook.sh /llm/start-notebook.sh

# Update the software sources
RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
# Install essential packages
    apt install software-properties-common libunwind8-dev vim less -y && \
# Install git, curl, and wget
    apt-get install -y git curl wget && \
# Install Python 3.11
    # Add Python 3.11 PPA repository
    add-apt-repository ppa:deadsnakes/ppa -y && \
    # Install Python 3.11
    apt-get install -y python3.11 && \
    # Remove the original /usr/bin/python3 symbolic link
    rm /usr/bin/python3 && \
    # Create a symbolic link pointing to Python 3.11 at /usr/bin/python3
    ln -s /usr/bin/python3.11 /usr/bin/python3 && \
    # Create a symbolic link pointing to /usr/bin/python3 at /usr/bin/python
    ln -s /usr/bin/python3 /usr/bin/python && \
    # Install Python 3.11 development and utility packages
    apt-get install -y python3-pip python3.11-dev python3-wheel python3.11-distutils && \
# Download and install pip, install FastChat from source requires PEP 660 support
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py && \
    pip install --upgrade requests argparse urllib3 && \
# Download ipex-llm-tutorial
    cd /llm && \
    pip install --upgrade jupyterlab && \
    git clone https://github.com/intel-analytics/ipex-llm-tutorial && \
    chmod +x /llm/start-notebook.sh && \
# Download all-in-one benchmark
    git clone https://github.com/intel-analytics/IPEX-LLM && \
    cp -r ./IPEX-LLM/python/llm/dev/benchmark/ ./benchmark && \
# Copy chat.py script
    pip install --upgrade colorama && \
    cp -r ./IPEX-LLM/python/llm/portable-zip/ ./portable-zip && \
# Install all-in-one dependencies
    apt-get install -y numactl && \
    pip install --upgrade omegaconf && \
    pip install --upgrade pandas && \
# Install vllm dependencies
    pip install --upgrade fastapi && \
    pip install --upgrade "uvicorn[standard]" && \
# Add Qwen support
    pip install --upgrade transformers_stream_generator einops && \
# Copy vLLM-Serving
    cp -r ./IPEX-LLM/python/llm/example/CPU/vLLM-Serving/ ./vLLM-Serving && \
    rm -rf ./IPEX-LLM && \
# Fix vllm service 
    pip install pydantic==1.10.11 && \
# Install ipex-llm
    cd /llm && \
    pip install --pre --upgrade ipex-llm[all] && \
    # Fix CVE-2024-22195
    pip install Jinja2==3.1.3 && \
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install intel-extension-for-pytorch==2.2.0 && \
    pip install oneccl_bind_pt==2.2.0 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/ && \
    pip install transformers==4.36.2
    
ENTRYPOINT ["/bin/bash"]
