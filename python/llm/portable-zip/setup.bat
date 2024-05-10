:: download python and extract zip
powershell -Command "Invoke-WebRequest https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip -OutFile python-embed-amd64.zip"
powershell -Command "Expand-Archive .\python-embed-amd64.zip -DestinationPath .\python-embed"
del .\python-embed-amd64.zip

set "PYTHONEXE=.\python-embed\python.exe"

:: download get-pip.py and install
powershell -Command "Invoke-WebRequest https://bootstrap.pypa.io/get-pip.py -OutFile .\python-embed\get-pip.py"
%PYTHONEXE% .\python-embed\get-pip.py

:: enable run site.main() automatically
cd .\python-embed
set "search=#import site"
set "replace=import site"
powershell -Command "(gc python311._pth) -replace '%search%', '%replace%' | Out-File -encoding ASCII python311._pth"
cd ..

:: install pip packages
if "%1"=="--gpu" (
    %PYTHONEXE% -m pip install dpcpp-cpp-rt==2024.0.2 mkl-dpcpp==2024.0.0 onednn==2024.0.0
    %PYTHONEXE% -m pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

    :: install libuv
    powershell -Command "Invoke-WebRequest https://anaconda.org/anaconda/libuv/1.44.2/download/win-64/libuv-1.44.2-h2bbff1b_0.tar.bz2 -OutFile libuv.tar.bz2"
    powershell -Command "Invoke-WebRequest https://www.7-zip.org/a/7za920.zip -OutFile 7za.zip"
    powershell -Command "Expand-Archive .\7za.zip -DestinationPath .\7za"
    del .\7za.zip
    .\7za\7za.exe x .\libuv.tar.bz2
    del .\libuv.tar.bz2
    .\7za\7za.exe x "-o./libuv" .\libuv.tar
    del .\libuv.tar

    :: copy libuv
    xcopy .\libuv\Library\* .\python-embed\Library /y /e /i /q

    rmdir /S /Q 7za
    rmdir /S /Q libuv

    copy chat-gpu.bat chat.bat
) else (
    %PYTHONEXE% -m pip install --pre --upgrade ipex-llm[all]

    :: modify chat.bat
    copy chat-cpu.bat chat.bat
)

%PYTHONEXE% -m pip install transformers_stream_generator tiktoken einops colorama

if "%1"=="--ui" (
    %PYTHONEXE% -m pip install --pre --upgrade ipex-llm[serving]
)

:: compress the python and scripts
if "%1"=="--ui" (
    powershell -Command "Compress-Archive -Path '.\python-embed', '.\chat-ui.bat', '.\kv_cache.py', '.\README.md' -DestinationPath .\ipex-llm-ui.zip"
) else (
    powershell -Command "Compress-Archive -Path '.\python-embed', '.\chat.bat', '.\chat.py', '.\kv_cache.py', '.\README.md' -DestinationPath .\ipex-llm.zip"
)