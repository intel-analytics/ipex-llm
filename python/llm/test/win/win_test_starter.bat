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

@REM Usage: call win_test_starter.bat %conda_activate_path% %base_dir%
@REM Example: win_test_starter.bat C:\ProgramData\Anaconda3\Scripts\activate.bat C:\Users\obe\bigdl-llm-test

set conda_activate_path=%1
set base_dir=%2

call %conda_activate_path% bigdl-llm
python %base_dir%\BigDL\python\llm\test\win\win_test_log.py --logger_dir %base_dir%\logs
set logger_file=%base_dir%\logs\win_llm_test.log
call %base_dir%\BigDL\python\llm\test\win\win_env_setup_and_test.bat %base_dir% >> %logger_file% 2>&1

pause

exit 0
