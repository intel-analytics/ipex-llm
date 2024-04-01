export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export LLM_HOME=${ANALYTICS_ZOO_ROOT}/python/llm/src
export LLM_INFERENCE_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/llm/test/langchain_gpu

export VICUNA_7B_1_3_ORIGIN_PATH=${VICUNA_7B_1_3_ORIGIN_PATH}

set -e

echo ">>> Testing LangChain upstream unit test"
mkdir ${LLM_INFERENCE_TEST_DIR}/langchain_upstream
wget https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/community/tests/integration_tests/llms/test_bigdl_llm.py -P ${LLM_INFERENCE_TEST_DIR}/langchain_upstream
wget https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/community/tests/integration_tests/llms/test_ipex_llm.py -P ${LLM_INFERENCE_TEST_DIR}/langchain_upstream
sed -i "s,model_id=\"[^\"]*\",model_id=\"$VICUNA_7B_1_3_ORIGIN_PATH\",g" ${LLM_INFERENCE_TEST_DIR}/langchain_upstream/test_bigdl_llm.py
sed -i "s,model_id=\"[^\"]*\",model_id=\"$VICUNA_7B_1_3_ORIGIN_PATH\",g" ${LLM_INFERENCE_TEST_DIR}/langchain_upstream/test_ipex_llm.py
sed -i 's/langchain_community.llms.ipex_llm/langchain_community.llms/g' ${LLM_INFERENCE_TEST_DIR}/langchain_upstream/test_ipex_llm.py
python -m pytest -s ${LLM_INFERENCE_TEST_DIR}/langchain_upstream

echo ">>> Testing LangChain upstream ipynb"
wget https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/docs/integrations/llms/ipex_llm.ipynb -P ${LLM_INFERENCE_TEST_DIR}/langchain_upstream
mv ${LLM_INFERENCE_TEST_DIR}/langchain_upstream/ipex_llm.ipynb ${LLM_INFERENCE_TEST_DIR}/langchain_upstream/langchain_example.ipynb
bash ./apps/ipynb2py.sh ${LLM_INFERENCE_TEST_DIR}/langchain_upstream/langchain_example
sed -i '/^get_ipython/d' ${LLM_INFERENCE_TEST_DIR}/langchain_upstream/langchain_example.py
sed -i "s,model_id=\"[^\"]*\",model_id=\"$VICUNA_7B_1_3_ORIGIN_PATH\",g" ${LLM_INFERENCE_TEST_DIR}/langchain_upstream/langchain_example.py
python ${LLM_INFERENCE_TEST_DIR}/langchain_upstream/langchain_example.py
rm -rf ${LLM_INFERENCE_TEST_DIR}/langchain_upstream