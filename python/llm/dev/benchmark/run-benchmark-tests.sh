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

if [ ! -d $ORIGINAL_LLAMA2_PATH ]; then
    echo "Directory $ORIGINAL_LLAMA2_PATH not found. Downloading from FTP server..."
    wget -r -nH --no-verbose --cut-dirs=1 $LLM_FTP_URL/${ORIGINAL_LLAMA2_PATH:2} -P $LLM_DIR
fi

echo ">>> Testing LLAMA2 transformers API"
std=$(taskset -c 0-$((THREAD_NUM - 1)) python python/llm/dev/benchmark/pipelines/llama2_test.py --repo-id-or-model-path $ORIGINAL_LLAMA2_PATH)
echo "the output of the example is: " 
echo $std
if [[ ! $std == *"AI is a term"* ]]; then
    echo "The expected output is not met."
    return 1
fi