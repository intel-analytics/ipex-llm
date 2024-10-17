@REM set SYCL_CACHE_PERSISTENT=1
@REM set BIGDL_LLM_XMX_DISABLED=1

@REM set IPEX_LLM_QUANTIZE_KV_CACHE=
@REM set IPEX_LLM_COMPRESS_KV_CACHE=
@REM @REM set IPEX_LLM_PERFORMANCE_MODE=1
set PYTHONPATH=D:\yina\BigDL\python\llm\src
@REM set PYTHONPATH=

@REM call C:\Users\arda\miniforge3\Scripts\activate

set BIGDL_USE_NPU=1
@REM set IPEX_LLM_CPU_LM_HEAD=0

@REM set IPEX_LLM_N_SPLITS_LINEAR=28
@REM set IPEX_LLM_N_SPLITS_DOWN_PROJ=148
@REM set IPEX_LLM_N_SPLITS_LINEAR=
@REM set IPEX_LLM_N_SPLITS_DOWN_PROJ=

@REM python qwen.py --repo-id-or-model-path D:\llm-models\Qwen2-7B-Instruct --prompt "Write an essay about AI"
python qwen.py --repo-id-or-model-path D:\llm-models\Qwen2-7B-Instruct --prompt "Why is the sky blue?" --n-predict 250
