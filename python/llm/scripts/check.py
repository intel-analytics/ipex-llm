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

import sys


def check_python_version():
    py_version = sys.version.split()[0]
    lst = py_version.split(".")
    v1 = eval(lst[0])
    v2 = eval(lst[1])
    if v1!=3 or v2<9:
        print("Python Version must be higher than 3.9.x, please check python version. More details could be found in the README.md")
        return 1
    return 0

def check_transformer_version():
    try:
        import transformers
        print(f"transformers={transformers.__version__}")
    except:
        print("Transformers is not installed.")

def check_torch_version():
    try:
        import torch
        print(f"torch={torch.__version__}")
    except:
        print("PyTorch is not installed.")


def check_torch_version():
    try:
        import torch
        print(f"torch={torch.__version__}")
    except:
        print("PyTorch is not installed.")

def check_bigdl_version():
    import os
    if os.system("pip show bigdl-llm")!=0:
        print("BigDL is not installed")


def check_ipex_version():
    try:
        import intel_extension_for_pytorch as ipex
        print(f"ipex={ipex.__version__}")
    except:
        print("IPEX is not installed properly. ")

def main():
    if check_python_version()!=0:
        return -1
    print("-----------------------------------------------------------------")
    check_transformer_version()
    print("-----------------------------------------------------------------")
    check_torch_version()
    print("-----------------------------------------------------------------")
    check_bigdl_version()
    print("-----------------------------------------------------------------")
    check_ipex_version()



if __name__ == "__main__":
    main()