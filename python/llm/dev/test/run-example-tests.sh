if [ -z "$THREAD_NUM" ]; then
  THREAD_NUM=2
fi
export OMP_NUM_THREADS=$THREAD_NUM

######## LLAMA2
# transformers
export ORIGINAL_LLAMA2_PATH=./llm/Llama-2-7b-chat-hf/
if [ ! -d $ORIGINAL_LLAMA2_PATH ]; then
    echo "Directory $ORIGINAL_LLAMA2_PATH not found. Downloading from FTP server..."
    wget -r -nH --no-verbose --cut-dirs=1 $LLM_FTP_URL/${ORIGINAL_LLAMA2_PATH:2} -P $LLM_DIR
fi


std=$(taskset -c 0-$((THREAD_NUM - 1)) python python/llm/example/transformers/transformers_int4/llama2/generate.py --repo-id-or-model-path $ORIGINAL_LLAMA2_PATH)
echo "the output of the example is: " 
echo $std
if [[ ! $std == *"AI is a term"* ]]; then
    echo "The expected output is not met."
    return 1
fi
# transformers low-bit
std=$(taskset -c 0-$((THREAD_NUM - 1)) python python/llm/example/transformers/transformers_low_bit/transformers_low_bit_pipeline.py --repo-id-or-model-path $ORIGINAL_LLAMA2_PATH)
echo "the output of the example is: " 
echo $std
if [[ ! $std == *"But her parents were always telling her to stay close to home"* ]]; then
    echo "The expected output is not met."
    return 1
fi

std=$(taskset -c 0-$((THREAD_NUM - 1)) python python/llm/example/transformers/transformers_low_bit/transformers_low_bit_pipeline.py --repo-id-or-model-path $ORIGINAL_LLAMA2_PATH --low-bit sym_int5)
echo "the output of the example is: " 
echo $std
# if [[ ! $std == *"But her parents were always telling her to stay close to home"* ]]; then
#     echo "The expected output is not met."
#     return 1
# fi

std=$(taskset -c 0-$((THREAD_NUM - 1)) python python/llm/example/transformers/transformers_low_bit/transformers_low_bit_pipeline.py --repo-id-or-model-path $ORIGINAL_LLAMA2_PATH --low-bit sym_int8)
echo "the output of the example is: " 
echo $std
# if [[ ! $std == *"But her parents were always telling her to stay close to home"* ]]; then
#     echo "The expected output is not met."
#     return 1
# fi

std=$(taskset -c 0-$((THREAD_NUM - 1)) python python/llm/example/transformers/transformers_low_bit/transformers_low_bit_pipeline.py --repo-id-or-model-path $ORIGINAL_LLAMA2_PATH --low-bit asym_int4)
echo "the output of the example is: " 
echo $std
# if [[ ! $std == *"But her parents were always telling her to stay close to home"* ]]; then
#     echo "The expected output is not met."
#     return 1
# fi

std=$(taskset -c 0-$((THREAD_NUM - 1)) python python/llm/example/transformers/transformers_low_bit/transformers_low_bit_pipeline.py --repo-id-or-model-path $ORIGINAL_LLAMA2_PATH --low-bit asym_int5)
echo "the output of the example is: " 
echo $std
# if [[ ! $std == *"But her parents were always telling her to stay close to home"* ]]; then
#     echo "The expected output is not met."
#     return 1
# fi


########## ChatGLM2
export ORIGINAL_CHATGLM2_6B_PATH=./llm/chatglm2-6b/
if [ ! -d $ORIGINAL_CHATGLM2_PATH ]; then
    echo "Directory $ORIGINAL_CHATGLM2_PATH not found. Downloading from FTP server..."
    wget -r -nH --no-verbose --cut-dirs=1 $LLM_FTP_URL/${ORIGINAL_CHATGLM2_PATH:2} -P $LLM_DIR
fi


std=$(taskset -c 0-$((THREAD_NUM - 1)) python python/llm/example/transformers/transformers_int4/llama2/generate.py --repo-id-or-model-path $ORIGINAL_CHATGLM2_PATH)
echo "the output of the example is: " 
echo $std
if [[ ! $std == *"AI指的是人工智能"* ]]; then
    echo "The expected output is not met."
    return 1
fi




