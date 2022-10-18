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


# Required Dependecies

# Install OpenVINO
# ```bash
# pip install openvino-dev
# ```
# Download model
# The following command is recommended to be executed in same directory as this script
# ```bash
# omz_downloader --name resnet18-xnor-binary-onnx-0001 -o ./model
# ```

import numpy as np

if __name__ == "__main__":
    # use resnet18 model pretrained on ImageNet dataset for example
    model_path = "model/intel/resnet18-xnor-binary-onnx-0001/FP16-INT1/resnet18-xnor-binary-onnx-0001.xml"

    # prepare input data
    x = [np.random.randn(1,3,224,224) for i in range(5)]

    # async inference using Nano
    from bigdl.nano.openvino import OpenVINOModel
    ov_model = OpenVINOModel(model=model_path)
    async_results = ov_model.async_predict(x, num_requests=5)
    # async_predict returns list of output array
    for output_array in async_results:
        print(output_array.argmax(axis=1))
