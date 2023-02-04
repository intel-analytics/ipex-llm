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


def PytorchJITINT8Model(model, calib_data, q_config=None,
                        input_sample=None, channels_last=None,
                        thread_num=None, jit_strict=True,
                        jit_method=None):
    from .jit_int8_model import PytorchJITINT8Model
    return PytorchJITINT8Model(model, calib_data, q_config=q_config,
                               input_sample=input_sample, channels_last=channels_last,
                               thread_num=thread_num, jit_strict=jit_strict,
                               jit_method=jit_method)

def load_pytorchjitint8_model(path, model):
    from .jit_int8_model import PytorchJITINT8Model
    return PytorchJITINT8Model._load(path, model)
