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

import os

FILE_PATH = "../../../tutorial/inference/tensorflow/cifar10.py"


def test_tutorial_inference_quantization():
    current_work_dir = os.path.dirname(__file__)
    os.chdir(current_work_dir)
    code = os.popen("sed 's/epochs = 20/epochs = 1/g' " + FILE_PATH).readlines()
    code = ''.join(code)
    global_ = {"__name__": "__main__"}
    exec(code, global_)


if __name__ == "__main__":
    test_tutorial_inference_quantization()
