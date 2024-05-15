FROM intelanalytics/ipex-llm-xpu:2.1.0-SNAPSHOT

ARG http_proxy
ARG https_proxy

# Disable pip's cache behavior
ARG PIP_NO_CACHE_DIR=false


# Install Serving Dependencies
RUN cd /llm &&\
    # Install ipex-llm[serving] only will update ipex_llm source code without updating
    # bigdl-core-xe, which will lead to problems
    apt-get update && \
    apt-get install -y libfabric-dev wrk unzip && \
    pip install --pre --upgrade ipex-llm[xpu,serving] && \
    pip install transformers==4.37.0 gradio==4.19.2 && \
    # Install vLLM-v2 dependencies
    cd /llm && \
    git clone -b sycl_xpu https://github.com/analytics-zoo/vllm.git && \
    cd vllm && \
    pip install -r requirements-xpu.txt && \
    pip install --no-deps xformers && \
    VLLM_BUILD_XPU_OPS=1 pip install --no-build-isolation -v -e . && \
    pip install outlines==0.0.34 --no-deps && \
    pip install interegular cloudpickle diskcache joblib lark nest-asyncio numba scipy && \
    # For Qwen series models support
    pip install transformers_stream_generator einops tiktoken && \
    # Install text-generation-webui dependencies
    cd /llm && wget https://github.com/intel-analytics/text-generation-webui/archive/refs/heads/ipex-llm.zip && \
    unzip ipex-llm.zip && rm ipex-llm.zip && \
    cd text-generation-webui-ipex-llm && \
    pip install -r requirements_cpu_only.txt && \
    # Install Langchain-Chatchat dependencies
    cd /llm && wget https://github.com/intel-analytics/Langchain-Chatchat/archive/refs/heads/ipex-llm.zip && \
    unzip ipex-llm.zip && rm ipex-llm.zip && \
    cd Langchain-Chatchat-ipex-llm && \
    pip install -r requirements_ipex_llm.txt && \
    pip install -r requirements_api_ipex_llm.txt && \
    pip install -r requirements_webui.txt && \
    python copy_config_example.py

ADD ./offline_inference.py      /llm/vllm-examples/
ADD ./payload-1024.lua          /llm/vllm-examples/
ADD ./start-vllm-service.sh     /llm/vllm-examples/
ADD ./benchmark_throughput.py   /llm/vllm-examples/
ADD ./start-fastchat-service.sh /llm/fastchat-examples/

WORKDIR /llm/
