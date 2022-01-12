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

import warnings
from functools import partial

QUANTIZATION_BINDED_COMPONENTS = ['_quantized_model',
                                  '_quantized_model_up_to_date',
                                  '_forward_fx_quantize',
                                  '_fx_quantize_on_train',
                                  '_fx_quantize_on_fit_start']


def _forward_fx_quantize(self, *args):
    return self._quantized_model(*args)


def _fx_quantize_on_train(self, mode=True):
    self._quantized_model_up_to_date = False
    self._quantized_model = None
    self.forward = self._torch_forward


def _fx_quantize_on_fit_start(self):
    self._quantized_model_up_to_date = False
    self._quantized_model = None
    self.forward = self._torch_forward


def _fx_quantize_eval(self, quantize=False):
    if quantize:
        if self._quantized_model_up_to_date:
            self.forward = self._forward_fx_quantize
        else:
            raise RuntimeError("Please call trainer.quantize again since the quantized model is"
                               "not up-to-date")
    else:
        self.forward = self._torch_forward


def bind_quantize_methods(pl_model, q_model):
    # check conflicts
    for component in QUANTIZATION_BINDED_COMPONENTS:
        if component in dir(pl_model):
            warnings.warn(f"{component} method/property will be replaced. You may rename your"
                          " customized attributes or methods and call `trainer.quantize again `"
                          "to avoid being overwrite.")

    pl_model._quantized_model = q_model
    pl_model._quantized_model_up_to_date = True
    pl_model._forward_fx_quantize = partial(_forward_fx_quantize, pl_model)
    pl_model._fx_quantize_eval = partial(_fx_quantize_eval, pl_model)
    pl_model._fx_quantize_on_train = partial(_fx_quantize_on_train, pl_model)
    pl_model._fx_quantize_on_fit_start = partial(_fx_quantize_on_fit_start, pl_model)

    return pl_model
