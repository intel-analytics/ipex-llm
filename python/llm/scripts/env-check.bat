@echo off


python -V
if ERRORLEVEL 1 ( 
    echo No Python found! How to create environment could be found in the README.md
    goto:end
)
python check.py

echo -----------------------------------------------------------------
echo System Information
systeminfo
echo -----------------------------------------------------------------
xpu-smi.exe
if ERRORLEVEL 1 ( 
    echo xpu-smi is not installed properly. 
    goto:end
)
