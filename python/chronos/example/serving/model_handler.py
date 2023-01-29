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
# The code is adapted from: https://github.com/pytorch/serve
#

import torch
import os

# Create model object
model = None

def entry_point_function_name(data, context):
    """
    Works on data and context to create model object or process inference request.
    Following sample demonstrates how model object can be initialized for jit mode.
    Similarly you can do it for eager mode models.
    :param data: Input data for prediction
    :param context: context contains model server system properties
    :return: prediction output
    """
    global model

    if not data:
        manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        device = torch.device("cpu")

        # Read model serialize/pt file
        serialized_file = manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        model = torch.jit.load(model_pt_path)
    else:
        # infererence and return result
        data_input = torch.Tensor(list(map(lambda x:float(x),
                                  data[0].get("body").decode().split(",")))).reshape(1, 6, 1)
        return model.forward(data_input).tolist()
