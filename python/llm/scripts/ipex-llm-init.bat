@echo off
setlocal

:: Usage #############################
:: ipex-llm-init.bat
:: Example:
:: ipex-llm-init.bat --gpu --device Arc
:: ######################################

REM Call the argument parsing function
call :parse-args %*

:: Initialize default values
set ENABLE_GPU=0
set DEVICE=""

:enable-gpu
SET ENABLE_GPU=1
GOTO :eof

:disable-gpu
SET ENABLE_GPU=0
CALL :unset-gpu-envs
GOTO :eof

:unset-gpu-envs
SET SYCL_CACHE_PERSISTENT=
SET BIGDL_LLM_XMX_DISABLED=
GOTO :eof

:display-var
echo +++++ Env Variables +++++
echo Exported:
echo     ENABLE_GPU             = %ENABLE_GPU%
echo     SYCL_CACHE_PERSISTENT  = %SYCL_CACHE_PERSISTENT%
echo     BIGDL_LLM_XMX_DISABLED = %BIGDL_LLM_XMX_DISABLED%
echo +++++++++++++++++++++++++
GOTO :eof

:display-help
echo Usage: call ipex-llm-init.bat [--option]
echo.
echo ipex-llm-init is a tool to automatically configure and run the subcommand under
echo environment variables for accelerating IPEX-LLM.
echo.
echo Optional options:
echo     -h, --help                Display this help message and exit.
echo     -g, --gpu                 Enable GPU support
echo     --device ^<device_type^>    Specify the device type (Arc, iGPU)
GOTO :eof

:display-error
echo Invalid Option: -%1 1>&2
echo.
CALL :display-help
EXIT /B 1

:parse-args
IF "%~1"=="" GOTO args-done
IF "%~1"=="-h" GOTO display-help
IF "%~1"=="--help" GOTO display-help
IF "%~1"=="-g" CALL :enable-gpu & echo DEBUG: ENABLE_GPU enabled & SHIFT & GOTO parse-args
IF "%~1"=="--gpu" CALL :enable-gpu & echo DEBUG: ENABLE_GPU enabled & SHIFT & GOTO parse-args
IF "%~1"=="--device" (
    IF "%~2"=="" (
        echo Error: --device option requires a value.
        GOTO :eof
    )
    SET "DEVICE=%2"
    SHIFT
    SHIFT
    GOTO parse-args
)
CALL :display-error %1
EXIT /B 1

:args-done

:: Ensure -g and --device are used together or not at all
IF "%ENABLE_GPU%"=="1" (
    IF "%DEVICE%"=="" (
        echo Error: --device must be specified with -g
        GOTO display-help
    )
) ELSE IF NOT "%DEVICE%"=="" (
    echo Error: -g must be specified with --device
    GOTO display-help
)

IF "%ENABLE_GPU%"=="1" (
    echo DEBUG: ENABLE_GPU is enabled and DEVICE is %DEVICE%
    IF /I "%DEVICE%"=="Arc" (
        SET SYCL_CACHE_PERSISTENT=1
    ) ELSE IF /I "%DEVICE%"=="iGPU" (
        SET SYCL_CACHE_PERSISTENT=1
        SET BIGDL_LLM_XMX_DISABLED=1
    ) ELSE (
        echo Error: Invalid device type specified for GPU.
        echo.
        CALL :display-help
        EXIT /B 1
    )
) ELSE (
    CALL :unset-gpu-envs
)

CALL :display-var
echo Complete.

:end
endlocal
