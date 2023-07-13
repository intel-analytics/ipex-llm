FROM mcr.microsoft.com/windows:20H2-KB5016616-amd64

SHELL ["powershell", "-Command", "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue';"]

RUN mkdir C:\ppml

# install python
ADD https://www.python.org/ftp/python/3.9.2/python-3.9.2-amd64.exe C:\ppml\python.exe
RUN Start-Process C:\ppml\python.exe -Wait -NoNewWindow -PassThru -ArgumentList @( '/quiet', 'InstallAllUsers=1', 'TargetDi
r=C:\Python', 'PrependPath=1', 'Shortcuts=0', 'Include_doc=0', 'Include_pip=0', 'Include_test=0' )

# install pip
RUN setx /M PATH "%PATH%;C:\Python"; python -m ensurepip

# install bigdl-llm
ADD ./bigdl-llm-windows.whl C:\ppml\bigdl-llm-windows.whl
ADD ./llm-convert C:\ppml\llm-convert
RUN cd C:\ppml; python -m pip install bigdl-llm-windows.wheel

CMD ["Set-Alias -Name llm-convert \"python C:\\ppml\\llm-convert\"; Set-Alias -Name llm-cli -Value .\\C:\\python\\Scripts\\llm-cli.ps1; Set-Alias -Name llm-chat -Value .\\C:\\python\\Scripts\\llm-chat.ps1; Set-ExecutionPolicy RemoteSigned"]
