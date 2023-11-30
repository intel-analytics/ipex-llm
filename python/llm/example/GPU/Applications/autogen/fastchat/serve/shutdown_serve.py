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
# Copyright 2023 The FastChat team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""
Usage：
python shutdown_serve.py --down all
options: "all","controller","model_worker","openai_api_server"， `all` means to stop all related servers 
"""

import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument(
    "--down", choices=["all", "controller", "model_worker", "openai_api_server"]
)
args = parser.parse_args()
base_shell = "ps -eo user,pid,cmd|grep fastchat.serve{}|grep -v grep|awk '{{print $2}}'|xargs kill -9"
if args.down == "all":
    shell_script = base_shell.format("")
else:
    serve = f".{args.down}"
    shell_script = base_shell.format(serve)
print(f"execute shell cmd: {shell_script}")
subprocess.run(shell_script, shell=True, check=True)
print(f"{args.down} has been shutdown!")
