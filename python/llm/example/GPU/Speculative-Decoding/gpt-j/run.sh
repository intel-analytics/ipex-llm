source /opt/intel/oneapi/setvars.sh --force

export CONDA_PREFIX=/home/wangyishuo/miniconda3/envs/yina-llm-ipex21
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export ENABLE_SDP_FUSION=1

export PYTHONPATH="/home/wangyishuo/yina/BigDL/python/llm/src"

MODEL_PATH="/mnt/disk1/huggingface/hub/models--EleutherAI--gpt-j-6B/snapshots/47e169305d2e8376be1d31e765533382721b2cc1/"
python speculative.py --repo-id-or-model-path $MODEL_PATH
