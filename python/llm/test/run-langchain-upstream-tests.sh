export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export LLM_HOME=${ANALYTICS_ZOO_ROOT}/python/llm/src
export LLM_INFERENCE_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/llm/test/langchain_gpu
export TEST_BIGDLLLM_MODEL_IDS=${VICUNA_7B_1_3_ORIGIN_PATH}
export TEST_IPEXLLM_MODEL_IDS=${VICUNA_7B_1_3_ORIGIN_PATH}

# Use Windows style path when running on Windows
if [[ $RUNNER_OS == "Windows" ]]; then
  export ANALYTICS_ZOO_ROOT=$(cygpath -m ${ANALYTICS_ZOO_ROOT})
  export TEST_BIGDLLLM_MODEL_IDS=$(cygpath -m ${VICUNA_7B_1_3_ORIGIN_PATH})
  export TEST_IPEXLLM_MODEL_IDS=$(cygpath -m ${VICUNA_7B_1_3_ORIGIN_PATH})
fi

set -e

echo ">>> Testing LangChain upstream unit test"
cp ${ANALYTICS_ZOO_ROOT}/langchain_upstream/libs/community/tests/integration_tests/llms/test_bigdl_llm.py ${ANALYTICS_ZOO_ROOT}/langchain_upstream
cp ${ANALYTICS_ZOO_ROOT}/langchain_upstream/libs/community/tests/integration_tests/llms/test_ipex_llm.py ${ANALYTICS_ZOO_ROOT}/langchain_upstream

source ${ANALYTICS_ZOO_ROOT}/python/llm/test/run-llm-check-function.sh

pytest_check_error python -m pytest -s ${ANALYTICS_ZOO_ROOT}/langchain_upstream/test_ipex_llm.py

echo ">>> Testing LangChain upstream ipynb"
cp ${ANALYTICS_ZOO_ROOT}/langchain_upstream/docs/docs/integrations/llms/ipex_llm.ipynb ${ANALYTICS_ZOO_ROOT}/langchain_upstream/langchain_example.ipynb
bash ./apps/ipynb2py.sh ${ANALYTICS_ZOO_ROOT}/langchain_upstream/langchain_example
sed -i '/^get_ipython/d' ${ANALYTICS_ZOO_ROOT}/langchain_upstream/langchain_example.py
sed -i "s,model_id=\"[^\"]*\",model_id=\"$TEST_IPEXLLM_MODEL_IDS\",g" ${ANALYTICS_ZOO_ROOT}/langchain_upstream/langchain_example.py
sed -i 's|saved_lowbit_model_path = "./vicuna-7b-1.5-low-bit"|saved_lowbit_model_path = "./langchain_upstream/vicuna-7b-1.5-low-bit"|' ${ANALYTICS_ZOO_ROOT}/langchain_upstream/langchain_example.py
ipex_workaround_wrapper python ${ANALYTICS_ZOO_ROOT}/langchain_upstream/langchain_example.py
rm -rf ${ANALYTICS_ZOO_ROOT}/langchain_upstream
