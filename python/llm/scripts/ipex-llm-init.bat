@echo off
setlocal

:: Usage #############################
:: ipex-llm-init.bat
:: Example:
:: ipex-llm-init.bat
:: ######################################

CALL :main %*
GOTO :end


:initialize
CALL :unset-gpu-envs
SET MODE=
GOTO :eof



:unset-gpu-envs
SET ONEAPI_DEVICE_SELECTOR_=%ONEAPI_DEVICE_SELECTOR%
SET SYCL_CACHE_PERSISTENT_=%SYCL_CACHE_PERSISTENT%
SET BIGDL_LLM_XMX_DISABLED_=%BIGDL_LLM_XMX_DISABLED%
SET ZE_AFFINITY_MASK_=%ZE_AFFINITY_MASK%
endlocal
SET SYCL_CACHE_PERSISTENT=
SET BIGDL_LLM_XMX_DISABLED=
SET ZE_AFFINITY_MASK=
SET ONEAPI_DEVICE_SELECTOR=
setlocal
GOTO :eof

:display-var
echo +++++ Env Variables +++++
echo Exported:
echo     ONEAPI_DEVICE_SELECTOR = %ONEAPI_DEVICE_SELECTOR_%
echo     SYCL_CACHE_PERSISTENT  = %SYCL_CACHE_PERSISTENT_%
echo     BIGDL_LLM_XMX_DISABLED = %BIGDL_LLM_XMX_DISABLED_%
echo +++++++++++++++++++++++++
GOTO :eof

:display-help
echo Usage: call ipex-llm-init.bat [--option]
echo.
echo ipex-llm-init is a tool to automatically configure and run the subcommand under
echo environment variables for accelerating IPEX-LLM.
echo.
echo Optional options:
echo     -h, --help                Display this help message then exit.
::echo     -g, --gpu                 Enable GPU support
::echo     -c, --cpu                 Enable CPU support
echo.
::echo If no option is specified, it will guide you to choose one.
GOTO :eof


:display-error-invalid-option
echo Invalid Option: %1 1>&2
echo.
CALL :display-help
GOTO :eof


:display-error-too-many-options
echo Too Many Options: ipex-llm-init is not supposed to receive more than one option. 1>&2
Call :display-help
GOTO :eof


:parse-args
IF "%~1"=="" SET MOD=gpu & GOTO :eof
IF "%~1"=="-h" SET MOD=help & GOTO :check-too-many-options
IF "%~1"=="--help" SET MOD=help & GOTO :check-too-many-options
::IF "%~1"=="-g" SET MOD=gpu & echo DEBUG: ENABLE_GPU enabled & SHIFT & GOTO :check-too-many-options
::IF "%~1"=="--gpu" SET MOD=gpu & echo DEBUG: ENABLE_GPU enabled & SHIFT & GOTO :check-too-many-options
::IF "%~1"=="-c" SET MOD=cpu & echo DEBUG: ENABLE_GPU disabled & SHIFT & GOTO :check-too-many-options
::IF "%~1"=="--cpu" SET MOD=cpu & echo DEBUG: ENABLE_GPU disabled & SHIFT & GOTO :check-too-many-options
:: IF "%~1"=="-a" SET MOD=auto & echo DEBUG: try to auto detect device & SHIFT & GOTO :check-too-many-options
:: IF "%~1"=="--auto" SET MOD=auto & echo DEBUG: try to auto detect device & SHIFT & GOTO :check-too-many-options

CALL :display-error-invalid-option %1
EXIT /B 1

:check-too-many-options
IF "%~2"=="" GOTO :eof
CALL :display-error-too-many-options
EXIT /B 1



:main
CALL :initialize
CALL :parse-args %*

if %ERRORLEVEL% NEQ 0 EXIT /B %ERRORLEVEL%

IF %MOD%==help (
    CALL :display-help
    GOTO :eof
)



if "%TMP%"=="" (
    SET TMPFILEPATH=.\tmp_ipex-llm-init_bat.tmp
) ELSE (
    SET TMPFILEPATH=%TMP%\tmp_ipex-llm-init_bat.tmp
)

python %~pd0ipex-llm-init-support.py %MOD% %TMPFILEPATH% 2>nul

if %ERRORLEVEL% NEQ 0 (
    echo Failed to run ipex-llm-init-support.py.
    echo Please check the error message above.
    EXIT /B 1
)

SET ONEAPI_DEVICE_SELECTOR_=
SET SYCL_CACHE_PERSISTENT_=
SET BIGDL_LLM_XMX_DISABLED_=
SET ZE_AFFINITY_MASK_=


(
    SET /p ONEAPI_DEVICE_SELECTOR_=
    SET /p SYCL_CACHE_PERSISTENT_=
    SET /p BIGDL_LLM_XMX_DISABLED_=
) < %TMPFILEPATH%
del %TMPFILEPATH%




CALL :display-var
echo Complete.

GOTO :eof

:end
endlocal & (
    SET ONEAPI_DEVICE_SELECTOR=%ONEAPI_DEVICE_SELECTOR_%
    SET SYCL_CACHE_PERSISTENT=%SYCL_CACHE_PERSISTENT_%
    SET BIGDL_LLM_XMX_DISABLED=%BIGDL_LLM_XMX_DISABLED_%
    SET ZE_AFFINITY_MASK=%ZE_AFFINITY_MASK_%
) 

