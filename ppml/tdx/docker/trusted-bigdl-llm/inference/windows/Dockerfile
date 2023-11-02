FROM mcr.microsoft.com/windows:20H2-KB5016616-amd64

SHELL ["powershell", "-Command", "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue';"]

RUN mkdir C:\ppml

# install python
ADD https://www.python.org/ftp/python/3.9.2/python-3.9.2-amd64.exe C:\\ppml\\python.exe
RUN Start-Process C:\\ppml\\python.exe -Wait -NoNewWindow -PassThru -ArgumentList @( '/quiet', 'InstallAllUsers=1', 'TargetDir=C:\Python', 'PrependPath=1', 'Shortcuts=0', 'Include_doc=0', 'Include_pip=0', 'Include_test=0' )

ADD ./bigdl_llm-2.4.0.dev0-py3-none-any.whl C:\\ppml\\bigdl_llm-2.4.0.dev0-py3-none-any.whl
ADD ./llm-convert C:\\ppml\\llm-convert

# install pip and bigdl-llm
RUN setx /M PATH '%PATH%;C:\Python'; python -m ensurepip; python -m pip install C:\\ppml\\bigdl_llm-2.4.0.dev0-py3-none-any.whl

CMD ["Set-Alias -Name llm-convert -Value 'python C:\ppml\llm-convert\'; Set-Alias -Name llm-cli -Value 'C:\python\Scripts\llm-cli.ps1'; Set-Alias -Name llm-chat -Value 'C:\python\Scripts\llm-chat.ps1'; Set-ExecutionPolicy RemoteSigned"]
