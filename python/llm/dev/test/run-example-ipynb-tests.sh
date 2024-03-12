if [ -z "$THREAD_NUM" ]; then
  THREAD_NUM=2
fi
export OMP_NUM_THREADS=$THREAD_NUM
export LLM_INFERENCE_TEST_DIR=./llm/test/langchain_gpu

echo ">>> Testing LangChain ipynb"
mkdir ${LLM_INFERENCE_TEST_DIR}/tmp_wget_dir_ipynb
wget https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/docs/integrations/llms/bigdl.ipynb -P ${LLM_INFERENCE_TEST_DIR}/tmp_wget_dir_ipynb
mv ${LLM_INFERENCE_TEST_DIR}/tmp_wget_dir_ipynb/bigdl.ipynb ${LLM_INFERENCE_TEST_DIR}/tmp_wget_dir_ipynb/langchain_example.ipynb
bash ./apps/ipynb2py.sh ${LLM_INFERENCE_TEST_DIR}/tmp_wget_dir_ipynb/langchain_example
sed -i '/^get_ipython/d' ${LLM_INFERENCE_TEST_DIR}/tmp_wget_dir_ipynb/langchain_example.py
sed -i "s,model_id=\"[^\"]*\",model_id=\"$VICUNA_7B_1_3_ORIGIN_PATH\",g" ${LLM_INFERENCE_TEST_DIR}/tmp_wget_dir_ipynb/langchain_example.py
python ${LLM_INFERENCE_TEST_DIR}/tmp_wget_dir_ipynb/langchain_example.py
rm -rf ${LLM_INFERENCE_TEST_DIR}/tmp_wget_dir_ipynb