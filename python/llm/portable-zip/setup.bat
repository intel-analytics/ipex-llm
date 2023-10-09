:: download python and extract zip
powershell -Command "Start-BitsTransfer -Source https://www.python.org/ftp/python/3.9.13/python-3.9.13-embed-amd64.zip -Destination .\python\llm\portable-zip\python-3.9.13-embed-amd64.zip"
powershell -Command "Expand-Archive .\python\llm\portable-zip\python-3.9.13-embed-amd64.zip -DestinationPath .\python\llm\portable-zip\python-embed"
del .\python-3.9.13-embed-amd64.zip

set "python-embed=.\python\llm\portable-zip\python-embed\python.exe"

:: download get-pip.py and install
powershell -Command "Invoke-WebRequest https://bootstrap.pypa.io/get-pip.py -OutFile .\python\llm\portable-zip\python-embed\get-pip.py"
%python-embed% .\python\llm\portable-zip\python-embed\get-pip.py

:: enable run site.main() automatically
@REM cd .\python\llm\portable-zip\python-embed
set "search=#import site"
set "replace=import site"
powershell -Command "(gc .\python\llm\portable-zip\python-embed\python39._pth) -replace '%search%', '%replace%' | Out-File -encoding ASCII .\python\llm\portable-zip\python-embed\python39._pth"
@REM cd ..

:: install pip packages
%python-embed% -m pip install bigdl-llm[all] transformers_stream_generator tiktoken einops colorama

:: compress the python and scripts
powershell -Command "Compress-Archive -Path '.\python\llm\portable-zip\python-embed', '.\python\llm\portable-zip\chat.bat', '.\python\llm\portable-zip\chat.py', '.\python\llm\portable-zip\README.md' -DestinationPath .\python\llm\portable-zip\bigdl-llm.zip"
