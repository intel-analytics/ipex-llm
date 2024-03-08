:: download python and extract zip
@REM if "%1"=="--python-zip" (
@REM     powershell -Command "Expand-Archive .\cpython-embed-zip.zip -DestinationPath .\python-embed"
@REM ) else (
@REM     powershell -Command "Start-BitsTransfer -Source https://www.python.org/ftp/python/3.11.8/python-3.11.8-embed-amd64.zip -Destination python-3.11.8-embed-amd64.zip"
@REM     powershell -Command "Expand-Archive .\python-3.11.8-embed-amd64.zip -DestinationPath .\python-embed"
@REM     del .\python-3.12.2-embed-amd64.zip
@REM )

@REM powershell -Command "Start-BitsTransfer -Source https://www.python.org/ftp/python/3.11.8/python-3.11.8-embed-amd64.zip -Destination python-3.11.8-embed-amd64.zip"
powershell -Command "Expand-Archive .\python-3.11.8-embed-amd64.zip -DestinationPath .\python-embed"
del .\python-3.12.2-embed-amd64.zip

set "python-embed=.\python-embed\python.exe"

:: download get-pip.py and install
powershell -Command "Invoke-WebRequest https://bootstrap.pypa.io/get-pip.py -OutFile .\python-embed\get-pip.py"
%python-embed% .\python-embed\get-pip.py

:: enable run site.main() automatically
cd .\python-embed
set "search=#import site"
set "replace=import site"
powershell -Command "(gc python311._pth) -replace '%search%', '%replace%' | Out-File -encoding ASCII python311._pth"
cd ..

:: install pip packages
%python-embed% -m pip install --pre --upgrade bigdl-llm[all]
%python-embed% -m pip install transformers==4.36.2
%python-embed% -m pip install transformers_stream_generator tiktoken einops colorama

if "%1"=="--ui" (
    %python-embed% -m pip install --pre --upgrade bigdl-llm[serving]
)

:: compress the python and scripts
if "%1"=="--ui" (
    powershell -Command "Compress-Archive -Path '.\python-embed', '.\kv_cache.py', '.\chat-ui.bat', '.\README.md' -DestinationPath .\bigdl-llm-portable-ui.zip"
) else (
    powershell -Command "Compress-Archive -Path '.\python-embed', '.\kv_cache.py', '.\chat.bat', '.\chat.py', '.\README.md' -DestinationPath .\bigdl-llm-portable.zip"
)