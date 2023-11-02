# Performance tests usually use dedicated machines, see below to set env vars, e.g. model paths
# The following environment variables should be ready
# ORIGINAL_LLAMA2_PATH
# LLAMA2_BASELINE
# LLM_DIR

if [ -z "$THREAD_NUM" ]; then
  THREAD_NUM=2
fi
export OMP_NUM_THREADS=$THREAD_NUM

######## LLAMA2
# transformers

echo ">>> Testing LLAMA2 transformers API"
taskset -c 0-$((THREAD_NUM - 1)) python python/llm/dev/benchmark/pipelines/llama2_test.py --repo-id-or-model-path $LLAMA2_7B_ORIGIN_PATH

