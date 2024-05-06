source /opt/intel/oneapi/setvars.sh
export no_proxy=localhost
export FI_PROVIDER=tcp
export OMP_NUM_THREADS=8

export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export TORCH_LLM_ALLREDUCE=0
export CCL_DG2_ALLREDUCE=1 # For internal CCL

export MODEL_PATH=1
CCL_ROOT=./1ccl_dg2_allreduce_20240308  LD_LIBRARY_PATH=./1ccl_dg2_allreduce_20240308/src:./1ccl_dg2_allreduce_20240308/deps/mpi/lib:/opt/intel/oneapi/2024.0/lib \
CCL_ZE_IPC_EXCHANGE=sockets torchrun --standalone --nnodes=1 --nproc-per-node 2 pipeline_serving.py --repo-id-or-model-path $MODEL_PATH --low-bit fp8
