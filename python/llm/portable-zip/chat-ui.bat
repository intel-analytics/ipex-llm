@echo off
set /p modelpath="Please enter the model path: "

powershell -Command "& { $env:no_proxy='localhost,127.0.0.1'; Start-Process -FilePath PowerShell -ArgumentList '-Command', '& { .\python-embed\python.exe -m fastchat.serve.controller > zip_controller.log 2>&1 }' -NoNewWindow }"
timeout /t 1 /nobreak >nul 2>&1
:loop1
powershell -Command "$output = Get-Content zip_controller.log; if($null -eq $output -or !($output | Select-String -Pattern 'Uvicorn running on')) { exit 1 } else { exit 0 }"
if errorlevel 1 (
    timeout /t 1 /nobreak >nul 2>&1
    goto loop1
)
echo [1/3] Controller started successfully

powershell -Command "& { $env:no_proxy='localhost,127.0.0.1'; Start-Process -FilePath PowerShell -ArgumentList '-Command', '& { .\python-embed\python.exe -m ipex_llm.serving.fastchat.ipex_llm_worker --model-path %modelpath% --device cpu --low-bit "sym_int4" --trust-remote-code > zip_model_worker.log 2>&1 }' -NoNewWindow }"
timeout /t 1 /nobreak >nul 2>&1
:loop2
powershell -Command "$output = Get-Content zip_model_worker.log; if($null -eq $output -or !($output | Select-String -Pattern 'Uvicorn running on')) { exit 1 } else { exit 0 }"
if errorlevel 1 (
    timeout /t 1 /nobreak >nul 2>&1
    goto loop2
)
echo [2/3] Model worker started successfully

powershell -Command "& { $env:no_proxy='localhost,127.0.0.1'; Start-Process -FilePath PowerShell -ArgumentList '-Command', '& { .\python-embed\python.exe -m fastchat.serve.gradio_web_server > zip_web_server.log 2>&1 }' -NoNewWindow }"
timeout /t 1 /nobreak >nul 2>&1
:loop3
powershell -Command "$output = Get-Content zip_web_server.log; if($null -eq $output -or !($output | Select-String -Pattern 'Running on local URL')) { exit 1 } else { exit 0 }"
if errorlevel 1 (
    timeout /t 1 /nobreak >nul 2>&1
    goto loop3
)
echo [3/3] Web server started successfully

echo All service started. Visit 127.0.0.1:7860 in browser to chat.

timeout /t -1 /nobreak >nul 2>&1