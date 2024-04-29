@echo off

REM Check Python version
python -V
if ERRORLEVEL 1 ( 
    echo No Python found! Instructions on how to create an environment can be found in the README.md.
    goto:end
)
python check.py

echo -----------------------------------------------------------------
echo System Information
systeminfo
echo -----------------------------------------------------------------
xpu-smi discovery
if ERRORLEVEL 1 ( 
    echo xpu-smi is not installed properly. 
    goto:end
)

:end
