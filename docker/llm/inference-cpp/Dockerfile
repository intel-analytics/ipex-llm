FROM intel/oneapi-basekit:2025.0.2-0-devel-ubuntu22.04

# Set build arguments for proxy
ARG http_proxy
ARG https_proxy

# Set environment variables
ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    SYCL_CACHE_PERSISTENT=1

# Disable pip cache
ARG PIP_NO_CACHE_DIR=false

# Install dependencies and configure the environment
RUN set -eux && \
    \
    # Configure Intel OneAPI and GPU repositories
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/intel-oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    chmod 644 /usr/share/keyrings/intel-oneapi-archive-keyring.gpg && \
    rm /etc/apt/sources.list.d/intel-graphics.list && \
    wget -O- https://repositories.intel.com/graphics/intel-graphics.key | gpg --dearmor | tee /usr/share/keyrings/intel-graphics.gpg > /dev/null && \
    echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy arc" | tee /etc/apt/sources.list.d/intel.gpu.jammy.list && \
    chmod 644 /usr/share/keyrings/intel-graphics.gpg && \
    \
    # Update and install basic dependencies
    apt-get update && \
    apt-get install -y --no-install-recommends \
      curl wget git sudo libunwind8-dev vim less gnupg gpg-agent software-properties-common && \
    \
    # Set timezone
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    \
    # Install Python 3.11
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y --no-install-recommends python3.11 python3-pip python3.11-dev python3.11-distutils python3-wheel && \
    rm /usr/bin/python3 && ln -s /usr/bin/python3.11 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    \
    # Install pip and essential Python packages
    wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py && \
    python3 get-pip.py && rm get-pip.py && \
    pip install --upgrade requests argparse urllib3 && \
    pip install --pre --upgrade ipex-llm[cpp] && \
    pip install transformers==4.36.2 transformers_stream_generator einops tiktoken && \
    \
    # Fix Trivy CVE issues
    apt-get remove -y libze-dev libze-intel-gpu1 && \
    \
    # Install Intel GPU OpenCL Driver and Compute Runtime
    mkdir -p /tmp/gpu && cd /tmp/gpu && \
    echo "Downloading Intel Compute Runtime (24.52) for Gen12+..." && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-core-2_2.5.6+18417_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-opencl-2_2.5.6+18417_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu_1.6.32224.5_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd_24.52.32224.5_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/libigdgmm12_22.5.5_amd64.deb && \
    \
    echo "Downloading Legacy Compute Runtime (24.35) for pre-Gen12 support..." && \
    wget https://github.com/intel/compute-runtime/releases/download/24.35.30872.22/intel-level-zero-gpu-legacy1_1.3.30872.22_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.35.30872.22/intel-opencl-icd-legacy1_24.35.30872.22_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17537.20/intel-igc-core_1.0.17537.20_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17537.20/intel-igc-opencl_1.0.17537.20_amd64.deb && \
    \
    dpkg -i *.deb && rm -rf /tmp/gpu && \
    \
    # Install oneAPI Level Zero Loader
    mkdir /tmp/level-zero && cd /tmp/level-zero && \
    wget https://github.com/oneapi-src/level-zero/releases/download/v1.20.2/level-zero_1.20.2+u22.04_amd64.deb && \
    wget https://github.com/oneapi-src/level-zero/releases/download/v1.20.2/level-zero-devel_1.20.2+u22.04_amd64.deb && \
    dpkg -i *.deb && rm -rf /tmp/level-zero && \
    \
    # Clean up unnecessary dependencies to reduce image size
    find /usr/lib/python3/dist-packages/ -name 'blinker*' -exec rm -rf {} + && \
    rm -rf /root/.cache/Cypress && \
    \
    # Create necessary directories
    mkdir -p /llm/scripts

# Copy startup scripts
COPY ./start-llama-cpp.sh /llm/scripts/
COPY ./start-ollama.sh /llm/scripts/
COPY ./benchmark_llama-cpp.sh /llm/scripts/
