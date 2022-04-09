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
from functools import partial


def bind_openvino_methods(pl_model):
    from . import torch_funcs
    pl_model.export_openvino = partial(torch_funcs.export, pl_model)
    pl_model.eval_openvino = partial(torch_funcs.eval_openvino, pl_model)
    pl_model.exit_openvino = partial(torch_funcs.exit_openvino, pl_model)
    return pl_model


def export(model, input_sample=None, xml_path="model.xml"):
    from . import torch_funcs
    torch_funcs.export(model, input_sample, xml_path)
