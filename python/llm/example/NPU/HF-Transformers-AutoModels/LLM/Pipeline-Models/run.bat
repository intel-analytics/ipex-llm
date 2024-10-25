set PYTHONPATH=D:\yina\BigDL\python\llm\src
@REM set PYTHONPATH=

set BIGDL_USE_NPU=1

python llama.py --repo-id-or-model-path "D:\llm-models\Llama-2-7b-chat-hf" --quantization_group_size 128