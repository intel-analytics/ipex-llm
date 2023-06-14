#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


# Start directory should be BigDL\python\llm\test\win
# python .\win_test.py

import subprocess
import os
from win_test_log import manage_logs

log_path = manage_logs("../../../../../logs")
log = open(log_path, 'a')
subprocess.Popen(["win_env_setup_and_test.bat"], shell=True, stdout=log, stderr=log).wait()
