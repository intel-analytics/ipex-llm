export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export LLM_HOME=${ANALYTICS_ZOO_ROOT}/python/llm/src
export LLM_INFERENCE_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/llm/test/langchain_gpu
export TEST_BIGDLLLM_MODEL_IDS=${VICUNA_7B_1_3_ORIGIN_PATH}
export TEST_IPEXLLM_MODEL_IDS=${VICUNA_7B_1_3_ORIGIN_PATH}

langchain_version=$(pip list | grep '^langchain\s' | awk '{print "v"$2}')

set -e

echo ">>> Testing LangChain upstream unit test"
mkdir ${LLM_INFERENCE_TEST_DIR}/langchain_upstream
git clone -n https://github.com/langchain-ai/langchain.git ${LLM_INFERENCE_TEST_DIR}/langchain_upstream
full_match_tag=$(git -C ${LLM_INFERENCE_TEST_DIR}/langchain_upstream tag | grep -E "^${langchain_version}$")
if [ -n "$full_match_tag" ]; then
    git -C ${LLM_INFERENCE_TEST_DIR}/langchain_upstream checkout $full_match_tag
else
    git -C ${LLM_INFERENCE_TEST_DIR}/langchain_upstream checkout $(git -C ${LLM_INFERENCE_TEST_DIR}/langchain_upstream tag | grep -E "^${langchain_version}" | head -n 1)
fi
cp ${LLM_INFERENCE_TEST_DIR}/langchain_upstream/libs/community/tests/integration_tests/llms/test_bigdl_llm.py ${LLM_INFERENCE_TEST_DIR}/langchain_upstream
cp ${LLM_INFERENCE_TEST_DIR}/langchain_upstream/libs/community/tests/integration_tests/llms/test_ipex_llm.py ${LLM_INFERENCE_TEST_DIR}/langchain_upstream

python -m pytest -s ${LLM_INFERENCE_TEST_DIR}/langchain_upstream/test_bigdl_llm.py
python -m pytest -s ${LLM_INFERENCE_TEST_DIR}/langchain_upstream/test_ipex_llm.py

echo ">>> Testing LangChain upstream ipynb"
cp ${LLM_INFERENCE_TEST_DIR}/langchain_upstream/docs/docs/integrations/llms/ipex_llm.ipynb ${LLM_INFERENCE_TEST_DIR}/langchain_upstream/langchain_example.ipynb
bash ./apps/ipynb2py.sh ${LLM_INFERENCE_TEST_DIR}/langchain_upstream/langchain_example
sed -i '/^get_ipython/d' ${LLM_INFERENCE_TEST_DIR}/langchain_upstream/langchain_example.py
sed -i "s,model_id=\"[^\"]*\",model_id=\"$TEST_IPEXLLM_MODEL_IDS\",g" ${LLM_INFERENCE_TEST_DIR}/langchain_upstream/langchain_example.py
sed -i 's|saved_lowbit_model_path = "./vicuna-7b-1.5-low-bit"|saved_lowbit_model_path = "./langchain_upstream/vicuna-7b-1.5-low-bit"|' ${LLM_INFERENCE_TEST_DIR}/langchain_upstream/langchain_example.py
python ${LLM_INFERENCE_TEST_DIR}/langchain_upstream/langchain_example.py
rm -rf ${LLM_INFERENCE_TEST_DIR}/langchain_upstream
