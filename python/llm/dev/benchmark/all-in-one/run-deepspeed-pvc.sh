export ZE_AFFINITY_MASK="0,1" # specify the used GPU
NUM_GPUS=2 # number of used GPU
export MASTER_ADDR=127.0.0.1
export FI_PROVIDER=tcp
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets

export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so:${LD_PRELOAD}
basekit_root=/opt/intel/oneapi
source $basekit_root/setvars.sh --force
source $basekit_root/ccl/latest/env/vars.sh --force

export OMP_NUM_THREADS=$((56/$NUM_GPUS))
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export TORCH_LLM_ALLREDUCE=1
export BIGDL_IMPORT_IPEX=0
mpirun -np $NUM_GPUS --prepend-rank python run.py
