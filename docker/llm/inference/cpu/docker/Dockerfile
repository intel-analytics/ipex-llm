FROM ubuntu:22.04

ARG http_proxy
ARG https_proxy
ARG PIP_NO_CACHE_DIR=false
ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1

COPY ./start-notebook.sh /llm/start-notebook.sh

# Install PYTHON 3.9
RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt install software-properties-common libunwind8-dev vim less -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.9 git curl wget && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.9 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get install -y python3-pip python3.9-dev python3-wheel python3.9-distutils && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
# Install FastChat from source requires PEP 660 support
    python3 get-pip.py && \
    rm get-pip.py && \
    pip install --upgrade requests argparse urllib3 && \
    pip3 install --no-cache-dir --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --pre --upgrade bigdl-llm[all] && \
# Download bigdl-llm-tutorial
    cd /llm && \
    pip install --upgrade jupyterlab && \
    git clone https://github.com/intel-analytics/bigdl-llm-tutorial && \
    chmod +x /llm/start-notebook.sh && \
# Download all-in-one benchmark
    git clone https://github.com/intel-analytics/BigDL && \
    cp -r ./BigDL/python/llm/dev/benchmark/ ./benchmark && \
# Copy chat.py script
    pip install --upgrade colorama && \
    cp -r ./BigDL/python/llm/portable-zip/ ./portable-zip && \
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
    cp -r ./BigDL/python/llm/example/CPU/vLLM-Serving/ ./vLLM-Serving && \
    rm -rf ./BigDL && \
# Install miniconda
    curl -LO "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh && \
# Install environment for speculative
    /root/miniconda3/condabin/conda init bash && \
    /bin/bash -c "source /root/.bashrc" && \
    /root/miniconda3/condabin/conda create -n bigdl-speculative-py39 -y python=3.9
RUN cp /root/miniconda3/condabin/conda /usr/bin && \
    conda init bash && \
    . ~/.bashrc && \
    conda activate bigdl-speculative-py39 && \
    cd /llm && \
    pip3 install --no-cache-dir --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --pre --upgrade bigdl-llm[all] && \
    wget https://raw.githubusercontent.com/intel/intel-extension-for-pytorch/0c63936d7a6740679987920367ae2e0cdb375b2e/scripts/compile_bundle.sh && \
    sed -i 's/VER_IPEX=main/VER_IPEX=0c63936d7a6740679987920367ae2e0cdb375b2e/g' "compile_bundle.sh" && \
    bash compile_bundle.sh && \
    cd ./intel-extension-for-pytorch && \
    pip install -r requirements.txt && \
    pip install transformers==4.36.2 && \
    pip install transformers_stream_generator
RUN cd /llm && \
    rm -rf llvm-project llvm-release intel-extension-for-pytorch compile_bundle.sh && \
    echo "conda deactivate" >> /root/.bashrc
    
ENTRYPOINT ["/bin/bash"]