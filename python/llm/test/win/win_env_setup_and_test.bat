@REM #
@REM # Copyright 2016 The BigDL Authors.
@REM #
@REM # Licensed under the Apache License, Version 2.0 (the "License");
@REM # you may not use this file except in compliance with the License.
@REM # You may obtain a copy of the License at
@REM #
@REM #     http://www.apache.org/licenses/LICENSE-2.0
@REM #
@REM # Unless required by applicable law or agreed to in writing, software
@REM # distributed under the License is distributed on an "AS IS" BASIS,
@REM # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@REM # See the License for the specific language governing permissions and
@REM # limitations under the License.
@REM #

@REM Usage: Start directory should be BigDL\python\llm\test\win
@REM python .\win_test.py

echo The current directory is %CD%

@REM Pull the latest code
@REM cd C:\Users\obe\bigdl-llm-test\BigDL
cd ..\..\..\..\
git pull

@REM Build and install bigdl-llm
pip uninstall bigdl-llm -y
pip uninstall numpy torch transformers sentencepiece accelerate -y
@REM cd C:\Users\obe\bigdl-llm-test\BigDL\python\llm
cd python\llm
pip install .[all] --use-pep517

@REM Run pytest
mkdir converted_models
@REM python C:\Users\obe\bigdl-llm-test\BigDL\python\llm\test\win\test_llama.py
python .\test\win\test_llama.py

@REM Clean up
pip uninstall bigdl-llm -y
pip uninstall numpy torch transformers sentencepiece accelerate -y
echo "Removing the quantized models and libs..."
rmdir /s /q converted_models
rmdir /s /q C:\Users\obe\bigdl-llm-test\BigDL\python\llm\src\bigdl\llm\libs

@REM Upload the log file
echo "Uploading the test logs to ftp..."
@REM ftp -s:C:\Users\obe\bigdl-llm-test\ftp.txt
ftp -s:..\..\..\ftp.txt

exit 0
