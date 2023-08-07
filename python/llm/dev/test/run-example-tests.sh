# LLAMA2 example test
if [ ! -d $ORIGINAL_LLAMA2_PATH ]; then
    echo "Directory $ORIGINAL_LLAMA2_PATH not found. Downloading from FTP server..."
    wget -r -nH --no-verbose --cut-dirs=1 $LLM_FTP_URL/${ORIGINAL_LLAMA2_PATH:2} -P $LLM_DIR
fi

source $CONDA_HOME/bin/activate bigdl-llm-test
std=$(python python/llm/example/transformers/transformers_int4/llama2/generate.py --repo-id-or-model-path $ORIGINAL_LLAMA2_PATH)
echo "the output of the example is: " 
echo $std
if [[ ! $std == *"AI is a term"* ]]; then
    echo "The expected output is not met."
    return 1
fi
source $CONDA_HOME/bin/deactivate