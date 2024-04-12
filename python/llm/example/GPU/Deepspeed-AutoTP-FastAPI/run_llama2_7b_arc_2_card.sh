export MASTER_ADDR=127.0.0.1
export FI_PROVIDER=tcp
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
export http_proxy=
export https_proxy=

export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so:${LD_PRELOAD}
basekit_root=/opt/intel/oneapi
source $basekit_root/setvars.sh --force
source $basekit_root/ccl/latest/env/vars.sh --force

NUM_GPUS=2 # number of used GPU
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export TORCH_LLM_ALLREDUCE=0

mpirun -np $NUM_GPUS --prepend-rank \
        python serving.py --repo-id-or-model-path YOUR_REPO_ID_OR_MODEL_PATH --low-bit 'sym_int4'

