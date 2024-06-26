FROM intelanalytics/ipex-llm-xpu:2.1.0-SNAPSHOT

ARG http_proxy
ARG https_proxy

# Disable pip's cache behavior
ARG PIP_NO_CACHE_DIR=false

# Install Serving Dependencies
# Install ipex-llm[serving] only will update ipex_llm source code without updating
# bigdl-core-xe, which will lead to problems
RUN apt-get update && \
    apt-get install -y --no-install-recommends libfabric-dev wrk libaio-dev && \
    pip install --pre --upgrade ipex-llm[xpu,serving] && \
    pip install transformers==4.37.0 gradio==4.19.2 && \
    # Install vLLM-v2 dependencies
    git clone -b sycl_xpu https://github.com/analytics-zoo/vllm.git /llm/vllm && \
    pip install -r /llm/vllm/requirements-xpu.txt && \
    pip install --no-deps xformers && \
    VLLM_BUILD_XPU_OPS=1 pip install --no-build-isolation -v -e /llm/vllm && \
    pip install outlines==0.0.34 --no-deps && \
    pip install interegular cloudpickle diskcache joblib lark nest-asyncio numba scipy && \
    # For Qwen series models support
    pip install transformers_stream_generator einops tiktoken && \
    # For pipeline serving support
    pip install mpi4py fastapi uvicorn openai && \
    pip install gradio # for gradio web UI && \
    git clone https://github.com/intel-analytics/ipex-llm /llm/ipex-llm && \
    mkdir -p /llm/pp_serving && \
    cp /llm/ipex-llm/python/llm/example/GPU/Pipeline-Parallel-FastAPI/*.py /llm/pp_serving

COPY ./vllm_offline_inference.py       /llm/
COPY ./payload-1024.lua                /llm/
COPY ./start-vllm-service.sh           /llm/
COPY ./benchmark_vllm_throughput.py   /llm/
COPY ./start-fastchat-service.sh       /llm/
COPY ./start-pp_serving-service.sh       /llm/


WORKDIR /llm/
