@echo off

:: execute chat script
set PYTHONUNBUFFERED=1

set SYCL_CACHE_PERSISTENT=1
set BIGDL_LLM_XMX_DISABLED=1

set path=%path%;%cd%\python-embed
set path=%path%;%cd%\python-embed\bin
set path=%path%;%cd%\python-embed\Library\bin
set path=%path%;%cd%\python-embed\Library\lib

set /p modelpath="Please enter the model path: "
.\python-embed\python.exe .\chat.py --model-path="%modelpath%" --device="gpu"

pause
