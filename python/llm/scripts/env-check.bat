@echo off


python -V
if ERRORLEVEL 1 echo No Python found! How to create environment could be found in the README.md

python check.py

echo -----------------------------------------------------------------
echo System Information
systeminfo

xpu-smi.exe
if ERRORLEVEL 1 echo xpu-smi is not installed properly. 