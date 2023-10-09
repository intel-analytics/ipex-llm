@echo off


:: execute chat script
set PYTHONUNBUFFERED=1

set /p modelpath="Please enter the model path: "
.\python-embed\python.exe .\chat.py --model-path="%modelpath%"

pause