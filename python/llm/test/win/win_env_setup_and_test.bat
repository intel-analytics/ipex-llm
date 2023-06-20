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
set base_dir=%1
echo %base_dir%

@REM Pull the latest code
cd %base_dir%\BigDL
git pull

@REM Build and install bigdl-llm
pip uninstall bigdl-llm -y
pip uninstall numpy torch transformers sentencepiece accelerate -y
pip install numpy torch transformers sentencepiece accelerate
pip install requests pytest
cd python\llm
@REM pip install .[all] --use-pep517
python setup.py clean --all install


@REM Run pytest
mkdir converted_models
set BLOOM_ORIGIN_PATH=%base_dir%\models\bloomz-7b1
set LLAMA_ORIGIN_PATH=%base_dir%\models\gpt4all-7b-hf
set GPTNEOX_ORIGIN_PATH=%base_dir%\models\gptneox-7b-redpajama-bf16
set INT4_CKPT_DIR=%base_dir%\converted_models
set LLAMA_INT4_CKPT_PATH=%INT4_CKPT_DIR%\bigdl_llm_llama_q4_0.bin
set GPTNEOX_INT4_CKPT_PATH=%INT4_CKPT_DIR%\bigdl_llm_gptneox_q4_0.bin
set BLOOM_INT4_CKPT_PATH=%INT4_CKPT_DIR%/bigdl_llm_bloom_q4_0.bin

echo "Running the convert models tests..."
python -m pytest -s .\test\convert\test_convert_model.py

echo "Running the inference models tests..."
python -m pytest -s .\test\inference\test_call_models.py

@REM Clean up
pip uninstall bigdl-llm -y
pip uninstall numpy torch transformers sentencepiece accelerate -y
echo "Removing the quantized models and libs..."
rmdir /s /q %INT4_CKPT_DIR%
rmdir /s /q %base_dir%\BigDL\python\llm\src\bigdl\llm\libs

@REM Upload the log file
echo "Uploading the test logs to ftp..."
ftp -s:..\..\..\ftp.txt

exit 0
