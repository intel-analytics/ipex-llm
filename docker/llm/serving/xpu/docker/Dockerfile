FROM intelanalytics/ipex-llm-serving-xpu:latest as build

ARG http_proxy
ARG https_proxy

ADD ./oneccl-binding.patch  /tmp/oneccl-binding.patch

RUN cd /tmp/ && \
    pip install --upgrade setuptools wheel twine && \
    pip install "setuptools<70.0.0" && \
    git clone https://github.com/intel/torch-ccl -b v2.1.100+xpu && \
    cd torch-ccl && \
    patch -p1 < /tmp/oneccl-binding.patch && \
    git submodule sync && \
    git submodule update --init --recursive && \
    COMPUTE_BACKEND=dpcpp python setup.py sdist bdist_wheel && \
    mv /tmp/torch-ccl/dist/oneccl_bind_pt-2.1.100+xpu-cp311-cp311-linux_x86_64.whl /tmp/


FROM intelanalytics/ipex-llm-xpu:2.2.0-SNAPSHOT

ARG http_proxy
ARG https_proxy

# Disable pip's cache behavior
ARG PIP_NO_CACHE_DIR=false
COPY --from=build /tmp/oneccl_bind_pt-2.1.100+xpu-cp311-cp311-linux_x86_64.whl /tmp/
ADD ./gradio_web_server.patch /tmp/gradio_web_server.patch

# Install Serving Dependencies
# Install ipex-llm[serving] only will update ipex_llm source code without updating
# bigdl-core-xe, which will lead to problems
RUN apt-get update && \
    apt-get install -y --no-install-recommends libfabric-dev wrk libaio-dev && \
    apt-get install -y intel-opencl-icd intel-level-zero-gpu=1.3.26241.33-647~22.04 level-zero level-zero-dev --allow-downgrades && \
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
    # for gradio web UI
    pip install gradio && \
    # Install internal oneccl && \
    cd /tmp/ && \
    wget https://sourceforge.net/projects/oneccl-wks/files/oneccl_wks_installer_2024.0.0.2.sh && \
    bash oneccl_wks_installer_2024.0.0.2.sh && \
    pip uninstall -y oneccl_bind_pt && \
    pip install /tmp/oneccl_bind_pt-2.1.100+xpu-cp311-cp311-linux_x86_64.whl && \ 
    rm /tmp/oneccl_bind_pt-2.1.100+xpu-cp311-cp311-linux_x86_64.whl && \
    patch /usr/local/lib/python3.11/dist-packages/fastchat/serve/gradio_web_server.py < /tmp/gradio_web_server.patch

COPY ./vllm_online_benchmark.py        /llm/
COPY ./vllm_offline_inference.py       /llm/
COPY ./payload-1024.lua                /llm/
COPY ./start-vllm-service.sh           /llm/
COPY ./benchmark_vllm_throughput.py   /llm/
COPY ./start-fastchat-service.sh       /llm/
COPY ./start-pp_serving-service.sh       /llm/
COPY ./start-lightweight_serving-service.sh       /llm/


WORKDIR /llm/
