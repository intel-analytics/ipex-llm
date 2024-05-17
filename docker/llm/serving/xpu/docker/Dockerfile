FROM intelanalytics/ipex-llm-xpu:2.1.0-SNAPSHOT as build

ARG http_proxy
ARG https_proxy

ADD ./oneccl-binding.patch  /tmp/oneccl-binding.patch

RUN cd /tmp/ && \
    pip install --upgrade setuptools wheel twine && \
    git clone https://github.com/intel/torch-ccl -b v2.1.100+xpu && \
    cd torch-ccl && \
    patch -p1 < /tmp/oneccl-binding.patch && \
    git submodule sync && \
    git submodule update --init --recursive && \
    COMPUTE_BACKEND=dpcpp python setup.py sdist bdist_wheel && \
    mv /tmp/torch-ccl/dist/oneccl_bind_pt-2.1.100+xpu-cp311-cp311-linux_x86_64.whl /tmp/

FROM intelanalytics/ipex-llm-xpu:2.1.0-SNAPSHOT

ARG http_proxy
ARG https_proxy

# Disable pip's cache behavior
ARG PIP_NO_CACHE_DIR=false

COPY --from=build /tmp/oneccl_bind_pt-2.1.100+xpu-cp311-cp311-linux_x86_64.whl /tmp/
ADD ./oneccl_wks_installer_rc1.sh /tmp/

# Install Serving Dependencies
RUN cd /llm &&\
    # Install ipex-llm[serving] only will update ipex_llm source code without updating
    # bigdl-core-xe, which will lead to problems
    apt-get update && \
    apt-get install -y libfabric-dev wrk && \
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
    pip uninstall -y oneccl_bind_pt && \
    pip install /tmp/oneccl_bind_pt-2.1.100+xpu-cp311-cp311-linux_x86_64.whl && \
    bash /tmp/oneccl_wks_installer_rc1.sh
    # TODO: cleanup

ADD ./vllm_offline_inference.py       /llm/
ADD ./payload-1024.lua                /llm/
ADD ./start-vllm-service.sh           /llm/
ADD ./benchmark_vllm_throughput.py    /llm/
ADD ./start-fastchat-service.sh       /llm/

WORKDIR /llm/
