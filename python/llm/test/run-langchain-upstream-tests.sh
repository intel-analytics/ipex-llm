export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export LLM_HOME=${ANALYTICS_ZOO_ROOT}/python/llm/src
export LLM_INFERENCE_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/llm/test/langchain_gpu

export VICUNA_7B_1_3_ORIGIN_PATH=${VICUNA_7B_1_3_ORIGIN_PATH}

set -e

rm -rf ${LLM_INFERENCE_TEST_DIR}/langchain_upstream
echo ">>> Testing LangChain upstream unit test"
mkdir ${LLM_INFERENCE_TEST_DIR}/langchain_upstream
wget https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/community/tests/integration_tests/llms/test_bigdl.py -P ${LLM_INFERENCE_TEST_DIR}/langchain_upstream
sed -i "s,model_id=\"[^\"]*\",model_id=\"$VICUNA_7B_1_3_ORIGIN_PATH\",g" ${LLM_INFERENCE_TEST_DIR}/langchain_upstream/test_bigdl.py
python -m pytest -s ${LLM_INFERENCE_TEST_DIR}/langchain_upstream

echo ">>> Testing LangChain upstream ipynb"
wget https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/docs/integrations/llms/bigdl.ipynb -P ${LLM_INFERENCE_TEST_DIR}/langchain_upstream
mv ${LLM_INFERENCE_TEST_DIR}/langchain_upstream/bigdl.ipynb ${LLM_INFERENCE_TEST_DIR}/langchain_upstream/langchain_example.ipynb
bash ./apps/ipynb2py.sh ${LLM_INFERENCE_TEST_DIR}/langchain_upstream/langchain_example
sed -i '/^get_ipython/d' ${LLM_INFERENCE_TEST_DIR}/langchain_upstream/langchain_example.py
sed -i "s,model_id=\"[^\"]*\",model_id=\"$VICUNA_7B_1_3_ORIGIN_PATH\",g" ${LLM_INFERENCE_TEST_DIR}/langchain_upstream/langchain_example.py
python ${LLM_INFERENCE_TEST_DIR}/langchain_upstream/langchain_example.py
rm -rf ${LLM_INFERENCE_TEST_DIR}/langchain_upstream