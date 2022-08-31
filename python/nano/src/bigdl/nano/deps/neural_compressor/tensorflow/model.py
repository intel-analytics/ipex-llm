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
from bigdl.nano.utils.inference.tf.model import AcceleratedKerasModel


class KerasQuantizedModel(AcceleratedKerasModel):

    def __init__(self, model):
        super().__init__(model)
        self._input = model.input_tensor
        self._output = model.output_tensor
        self._sess = model.sess

    def on_forward_start(self, inputs):
        return self.tensors_to_numpy(inputs)

    def forward_step(self, *inputs):
        input_dict = dict(zip(self._input, inputs))
        out = self._sess.run(self._output, feed_dict=input_dict)
        return out

    def on_forward_end(self, outputs):
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    def _save_model(self, path):
        return super()._save_model(path)

    @staticmethod
    def _load(path, model=None):
        return super()._load(path)
