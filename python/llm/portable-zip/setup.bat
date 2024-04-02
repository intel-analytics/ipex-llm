:: download python and extract zip
powershell -Command "Start-BitsTransfer -Source https://www.python.org/ftp/python/3.9.13/python-3.9.13-embed-amd64.zip -Destination python-3.9.13-embed-amd64.zip"
powershell -Command "Expand-Archive .\python-3.9.13-embed-amd64.zip -DestinationPath .\python-embed"
del .\python-3.9.13-embed-amd64.zip

set "python-embed=.\python-embed\python.exe"

:: download get-pip.py and install
powershell -Command "Invoke-WebRequest https://bootstrap.pypa.io/get-pip.py -OutFile .\python-embed\get-pip.py"
%python-embed% .\python-embed\get-pip.py

:: enable run site.main() automatically
cd .\python-embed
set "search=#import site"
set "replace=import site"
powershell -Command "(gc python39._pth) -replace '%search%', '%replace%' | Out-File -encoding ASCII python39._pth"
cd ..

:: install pip packages
%python-embed% -m pip install --pre --upgrade ipex-llm[all]
%python-embed% -m pip install transformers_stream_generator tiktoken einops colorama

if "%1"=="--ui" (
    %python-embed% -m pip install --pre --upgrade ipex-llm[serving]
)

:: compress the python and scripts
if "%1"=="--ui" (
    powershell -Command "Compress-Archive -Path '.\python-embed', '.\chat-ui.bat', '.\README.md' -DestinationPath .\ipex-llm-ui.zip"
) else (
    powershell -Command "Compress-Archive -Path '.\python-embed', '.\chat.bat', '.\chat.py', '.\README.md' -DestinationPath .\ipex-llm.zip"
)